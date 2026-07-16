#!/usr/bin/env python3
"""Train an anomaly detection model on normal-only data.

Trains Isolation Forest on the training set (assumed mostly normal), computes
anomaly scores on train+test, selects a threshold, and outputs all artefacts.

Usage:
    uv run python pdm/anomaly_detection/train_anomaly.py --train ./data/raw_train.csv --test ./data/raw_test.csv
    uv run python pdm/anomaly_detection/train_anomaly.py --train ./data/raw_train.csv --test ./data/raw_test.csv --contamination 0.05
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib


def _select_features(df: pd.DataFrame) -> list[str]:
    """Select numeric feature columns (exclude label/target/id columns)."""
    exclude_prefixes = ("label_", "target", "anomaly", "health")
    exclude_exact = {"RUL", "machine_failure", "duration", "event", "unit_id", "device_id", "observation_date"}
    cols = []
    for c in df.select_dtypes(include=[np.number]).columns:
        if c in exclude_exact:
            continue
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        cols.append(c)
    return cols


def _detect_normal_mask(df: pd.DataFrame) -> pd.Series:
    """Detect normal samples. If labels exist, normal = all labels 0. Otherwise assume all normal."""
    label_cols = [c for c in df.columns if c.startswith("label_")]
    if label_cols:
        return df[label_cols].sum(axis=1) == 0
    if "machine_failure" in df.columns:
        return df["machine_failure"] == 0
    if "anomaly" in df.columns:
        return df["anomaly"] == 0
    # No labels → assume all training data is normal (standard AD assumption)
    return pd.Series(True, index=df.index)


def train(args):
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    feature_cols = _select_features(train_df)
    if not feature_cols:
        sys.exit("Error: no numeric feature columns found")

    print(f"{'='*60}")
    print("  Anomaly Detection — Isolation Forest")
    print(f"  Features: {len(feature_cols)} | Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"  Contamination: {args.contamination}")
    print(f"{'='*60}\n")

    # Filter to normal-only training data
    normal_mask = _detect_normal_mask(train_df)
    train_normal = train_df.loc[normal_mask, feature_cols].copy()
    n_excluded = (~normal_mask).sum()
    if n_excluded > 0:
        print(f"  Excluded {n_excluded} labeled-anomaly rows from training ({n_excluded/len(train_df):.1%})")

    # Drop high-NaN and zero-variance features
    nan_rate = train_normal.isna().mean()
    high_nan = nan_rate[nan_rate > 0.5].index.tolist()
    train_normal = train_normal.drop(columns=high_nan)
    feature_cols = [c for c in feature_cols if c not in high_nan]

    train_normal = train_normal.fillna(0)
    variances = train_normal.var()
    zero_var = variances[variances == 0].index.tolist()
    train_normal = train_normal.drop(columns=zero_var)
    feature_cols = [c for c in feature_cols if c not in zero_var]

    if high_nan or zero_var:
        print(f"  Dropped {len(high_nan)} high-NaN + {len(zero_var)} zero-variance features → {len(feature_cols)} remaining")

    print(f"  Training on {len(train_normal)} normal samples\n")

    # Scale and train
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_normal)

    # Train Isolation Forest
    max_features = min(1.0, max(0.1, 50 / len(feature_cols)))  # cap feature sampling for high-dim
    model = IsolationForest(
        n_estimators=args.n_estimators,
        max_features=max_features,
        contamination=args.contamination,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(X_train)
    print(f"  max_features: {max_features:.3f} ({int(max_features * len(feature_cols))} per tree)")

    # Score both sets (negative = more anomalous in sklearn convention)
    X_test = scaler.transform(test_df[feature_cols].fillna(0))
    train_scores = -model.score_samples(X_train)  # flip: higher = more anomalous
    test_scores = -model.score_samples(X_test)

    # Threshold: percentile on training scores
    threshold = float(np.percentile(train_scores, (1 - args.contamination) * 100))
    print(f"  Threshold (p{(1-args.contamination)*100:.0f} on train): {threshold:.6f}")
    print(f"  Train anomaly rate: {(train_scores > threshold).mean():.2%}")
    print(f"  Test anomaly rate:  {(test_scores > threshold).mean():.2%}")

    # Save artefacts
    # Ensure output is a Path (supports both str and Path when called programmatically)
    args.output = Path(args.output) if not isinstance(args.output, Path) else args.output
    args.output.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, args.output / "isolation_forest.joblib")
    joblib.dump(scaler, args.output / "scaler.joblib")

    # Scores CSV
    scores_df = test_df[["device_id"] if "device_id" in test_df.columns else []].copy()
    if "observation_date" in test_df.columns:
        scores_df["observation_date"] = test_df["observation_date"]
    scores_df["anomaly_score"] = test_scores
    scores_df["is_anomaly"] = (test_scores > threshold).astype(int)
    scores_df.to_csv(args.output / "anomaly_scores_test.csv", index=False)

    # Threshold + config
    threshold_data = {
        "threshold": threshold,
        "strategy": f"percentile_{(1-args.contamination)*100:.0f}",
        "contamination": args.contamination,
        "train_anomaly_rate": float((train_scores > threshold).mean()),
        "test_anomaly_rate": float((test_scores > threshold).mean()),
    }
    (args.output / "threshold.json").write_text(json.dumps(threshold_data, indent=2))

    # Metadata
    metadata = {
        "model_type": "IsolationForest",
        "feature_names": feature_cols,
        "n_estimators": args.n_estimators,
        "contamination": args.contamination,
        "n_train_normal": len(train_normal),
        "n_train_excluded": int(n_excluded),
        "threshold": threshold,
        "training_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Baseline distributions for drift monitoring
    baseline = {}
    for col in feature_cols:
        s = train_normal[col]
        baseline[col] = {
            "mean": float(s.mean()), "std": float(s.std()),
            "q25": float(s.quantile(0.25)), "q50": float(s.quantile(0.5)),
            "q75": float(s.quantile(0.75)),
        }
    (args.output / "baseline_stats.json").write_text(json.dumps(baseline, indent=2))

    print(f"\n✅ Saved to {args.output}/")
    return train_scores, test_scores, threshold, feature_cols, metadata


def main():
    parser = argparse.ArgumentParser(description="Train anomaly detection model (Isolation Forest)")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("./anomaly_detection/model"))
    parser.add_argument("--contamination", type=float, default=0.05, help="Expected anomaly fraction (default 5%%)")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for p in (args.train, args.test):
        if not p.exists():
            sys.exit(f"Error: {p} not found")

    train(args)


if __name__ == "__main__":
    main()
