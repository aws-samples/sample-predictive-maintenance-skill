#!/usr/bin/env python3
"""Train AutoML PdM model using AutoGluon.

Auto-detects formulation from dataset columns:
  - label_* columns → multi-label classification (one TabularPredictor per label)
  - RUL column → RUL regression
  - machine_failure → binary classification
  - duration + event → survival (regression on duration)

Usage:
    uv run python pdm/fault_prediction/train.py --train ./data/train.csv --test ./data/test.csv
    uv run python pdm/fault_prediction/train.py --train ./data/train.csv --test ./data/test.csv --time-limit 600
    uv run python pdm/fault_prediction/train.py --train ./data/train.csv --test ./data/test.csv --presets best
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import f1_score, precision_score, recall_score


def nasa_scoring(y_true, y_pred):
    """NASA PHM'08 asymmetric scoring function."""
    d = y_pred - y_true
    return float(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1).sum())


def _nasa_ag_metric(y_true, y_pred):
    """AutoGluon-compatible NASA scoring (lower is better, negated for maximization)."""
    d = y_pred - y_true
    score = float(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1).sum())
    return -score  # negate so AutoGluon maximizes


nasa_ag_scorer = None
try:
    from autogluon.core.metrics import make_scorer
    nasa_ag_scorer = make_scorer("nasa_score", _nasa_ag_metric, greater_is_better=True, needs_proba=False)
except ImportError:
    pass


def detect_formulation(df: pd.DataFrame) -> str:
    """Return: 'multilabel', 'regression', 'classification', or 'survival'."""
    label_cols = [c for c in df.columns if c.startswith("label_")]
    if label_cols:
        return "multilabel"
    if "RUL" in df.columns:
        return "regression"
    if "machine_failure" in df.columns:
        return "classification"
    if "duration" in df.columns and "event" in df.columns:
        return "survival"
    raise ValueError("Cannot detect formulation. Need label_* columns, 'RUL', 'machine_failure', or 'duration'+'event'")


def _append_progress(label, result, output_dir):
    """Append one label's result to training_progress.md for live monitoring."""
    progress_file = output_dir / "training_progress.md"
    if not progress_file.exists():
        progress_file.write_text("# Training Progress\n\n| Label | F1 | Precision | Recall | Pos Rate | Status |\n|-------|----|-----------|----|----------|--------|\n")
    if "test_f1" in result:
        line = f"| {label} | {result['test_f1']:.4f} | {result['test_precision']:.4f} | {result['test_recall']:.4f} | {result['positive_rate']:.3%} | ✅ |\n"
    else:
        line = f"| {label} | — | — | — | — | ⏭️ {result.get('reason', 'skipped')} |\n"
    with open(progress_file, "a") as f:
        f.write(line)


def _train_single_label(label, feature_cols, train_df, test_df, output_dir, time_limit, presets, rebalance=False, skip_importance=False):
    """Train one label — designed to be called in parallel."""
    pos_rate = train_df[label].mean()
    n_pos = int(train_df[label].sum())

    if n_pos < 10:
        return label, {"skipped": True, "reason": "too_few_positives"}, None

    train_label_df = train_df[feature_cols + [label]].copy()
    test_label_df = test_df[feature_cols + [label]]

    if rebalance and pos_rate < 0.5:
        weights = train_label_df[label].map({1: 1.0 / max(pos_rate, 0.01), 0: 1.0 / (1 - pos_rate)})
        train_label_df["sample_weight"] = weights / weights.mean()

    predictor = TabularPredictor(
        label=label, eval_metric="f1", problem_type="binary",
        path=str(output_dir / "ag_model" / label.replace(" ", "_")),
        verbosity=0,
    ).fit(train_data=train_label_df, time_limit=time_limit, presets=presets)

    y_pred = predictor.predict(test_label_df.drop(columns=[label]))
    y_true = test_label_df[label].values
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    importance_list = None
    top_feats = []
    if not skip_importance:
        try:
            importance = predictor.feature_importance(test_label_df, silent=True)
            top_feats = list(importance.head(5).index)
            importance_list = [
                {"feature": f, "importance": round(float(importance.loc[f, "importance"]), 6)}
                for f in importance.head(10).index
            ]
        except Exception:
            pass

    result = {
        "test_f1": round(f1, 4), "test_precision": round(prec, 4),
        "test_recall": round(rec, 4), "positive_rate": round(pos_rate, 5),
        "n_positives_train": n_pos, "best_model": predictor.model_best,
        "top_features": top_feats,
    }
    return label, result, importance_list


def train_multilabel(train_df, test_df, args):
    """Train one TabularPredictor per label_* column (parallel when --workers > 1)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    label_cols = sorted([c for c in train_df.columns if c.startswith("label_")])
    feature_cols = [c for c in train_df.columns if c not in label_cols]

    print(f"  Labels: {len(label_cols)} | Features: {len(feature_cols)}")
    print(f"  Time/label: {args.time_limit}s | Presets: {args.presets} | Workers: {args.workers}\n")

    metrics = {"formulation": "multi-label classification", "labels": {}}
    all_importance = {}

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _train_single_label, label, feature_cols, train_df, test_df,
                    args.output, args.time_limit, args.presets, args.rebalance,
                    args.skip_importance
                ): label for label in label_cols
            }
            for future in as_completed(futures):
                label, result, importance_list = future.result()
                metrics["labels"][label] = result
                if importance_list:
                    all_importance[label] = importance_list
                status = f"F1={result['test_f1']}" if "test_f1" in result else "SKIPPED"
                print(f"  ✓ {label} — {status}")
                _append_progress(label, result, args.output)
    else:
        for i, label in enumerate(label_cols):
            print(f"[{i+1}/{len(label_cols)}] {label}")
            label, result, importance_list = _train_single_label(
                label, feature_cols, train_df, test_df,
                args.output, args.time_limit, args.presets, args.rebalance,
                args.skip_importance
            )
            metrics["labels"][label] = result
            if importance_list:
                all_importance[label] = importance_list
            status = f"F1={result['test_f1']}" if "test_f1" in result else "SKIPPED"
            print(f"  {status}")
            _append_progress(label, result, args.output)

    scored = {k: v for k, v in metrics["labels"].items() if "test_f1" in v}
    if scored:
        f1s = [v["test_f1"] for v in scored.values()]
        metrics["mean_test_f1"] = round(np.mean(f1s), 4)
        metrics["median_test_f1"] = round(np.median(f1s), 4)
        metrics["labels_above_050"] = sum(1 for f in f1s if f >= 0.50)
        metrics["labels_total_scored"] = len(scored)
    if all_importance:
        metrics["feature_importance"] = all_importance

    metadata = {
        "feature_names": feature_cols, "label_names": label_cols,
        "formulation": "multi-label classification",
        "training_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "n_train_samples": len(train_df), "n_features": len(feature_cols),
    }

    if all_importance:
        _plot_multilabel_importance(all_importance, metrics, args.output)

    return metrics, metadata


def train_single(train_df, test_df, formulation, args):
    """Train a single TabularPredictor for regression/classification/survival."""
    target_map = {"regression": "RUL", "classification": "machine_failure", "survival": "duration"}
    target_col = target_map[formulation]
    problem_type = "binary" if formulation == "classification" else "regression"
    eval_metric = "f1" if formulation == "classification" else "root_mean_squared_error"

    # Use NASA scoring as objective for RUL if available
    if formulation == "regression" and nasa_ag_scorer is not None:
        eval_metric = nasa_ag_scorer

    drop_cols = []
    if formulation == "survival":
        drop_cols = ["event"]
    elif formulation == "regression":
        drop_cols = [c for c in ["unit_id", "cycle"] if c in train_df.columns]
    if drop_cols:
        train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
        test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    print(f"  Target: {target_col} | Metric: {eval_metric}")

    if args.rebalance and formulation == "classification":
        pos_rate = train_df[target_col].mean()
        if pos_rate < 0.5:
            weights = train_df[target_col].map({1: 1.0 / max(pos_rate, 0.01), 0: 1.0 / (1 - pos_rate)})
            train_df = train_df.copy()
            train_df["sample_weight"] = weights / weights.mean()
            print(f"  Rebalance: applied inverse-frequency weights (pos_rate={pos_rate:.3%})")

    if args.smote and formulation == "classification":
        from imblearn.over_sampling import SMOTE
        feature_cols_smote = [c for c in train_df.columns if c != target_col and c != "sample_weight"]
        sm = SMOTE(random_state=args.seed)
        X_res, y_res = sm.fit_resample(train_df[feature_cols_smote], train_df[target_col])
        train_df = pd.concat([X_res, y_res], axis=1)
        print(f"  SMOTE: oversampled minority → {len(train_df)} rows (was {len(X_res) - (y_res.sum() - train_df[target_col].sum())})")

    predictor = TabularPredictor(
        label=target_col, eval_metric=eval_metric, problem_type=problem_type,
        path=str(args.output / "ag_model"),
    )
    if args.warm_start and (args.output / "ag_model").exists():
        print("  Warm-start: loading existing model and refitting...")
        predictor = TabularPredictor.load(str(args.output / "ag_model"))
        predictor.refit_full(train_df)
    else:
        predictor.fit(train_data=train_df, time_limit=args.time_limit, presets=args.presets)

    perf = predictor.evaluate(test_df)
    leaderboard = predictor.leaderboard(test_df, silent=True)
    print(f"\nLeaderboard:\n{leaderboard[['model', 'score_test']].head(5).to_string()}")

    # Ensemble diversity check
    model_types = leaderboard["model"].head(5).tolist()
    families = set(m.split("_")[0] for m in model_types)
    if len(families) == 1:
        print(f"  ⚠️ Low ensemble diversity: top 5 models are all {list(families)[0]} variants")

    importance = None
    feature_cols = [c for c in train_df.columns if c != target_col]

    if not args.skip_importance:
        importance = predictor.feature_importance(test_df, silent=True)

        # Bootstrap importance stability (5 samples)
        bootstrap_ranks = {}
        for _ in range(5):
            sample = test_df.sample(frac=0.8, replace=True)
            try:
                imp_b = predictor.feature_importance(sample, silent=True)
                for feat in imp_b.head(10).index:
                    bootstrap_ranks.setdefault(feat, []).append(int(list(imp_b.index).index(feat)))
            except Exception:
                break
        unstable = [f for f, ranks in bootstrap_ranks.items() if max(ranks) - min(ranks) > 10]
        if unstable:
            print(f"  ⚠️ Unstable feature rankings (>10 rank swing): {unstable[:5]}")

    metrics = {
        "formulation": formulation, "metric": eval_metric,
        "test_score": float(perf[eval_metric]) if isinstance(perf, dict) else float(perf),
        "best_model": predictor.model_best,
    }
    if importance is not None:
        metrics["feature_importance"] = importance.head(20)["importance"].to_dict()

    if formulation == "regression":
        y_pred = predictor.predict(test_df.drop(columns=[target_col]))
        y_true = test_df[target_col].values
        score = nasa_scoring(y_true, y_pred.values)
        metrics["nasa_score"] = score
        metrics["nasa_score_normalized"] = score / len(y_true)
        print(f"NASA Score: {score:.1f} (normalized: {score/len(y_true):.2f})")

    if formulation == "classification":
        y_pred = predictor.predict(test_df.drop(columns=[target_col]))
        y_true = test_df[target_col].values
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        print(f"  Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")

    metadata = {
        "feature_names": feature_cols, "target_name": target_col,
        "formulation": formulation,
        "training_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "n_train_samples": len(train_df), "n_features": len(feature_cols),
    }

    # Plot
    if importance is not None:
        top_feats = importance.head(15)
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        top_feats["importance"].plot.barh(ax=ax)
        ax.set_xlabel("Importance (permutation)")
        ax.set_title("Top 15 Feature Importances")
        plt.tight_layout()
        plt.savefig(args.output / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()

    return metrics, metadata


def _generate_model_card(metrics, metadata, args):
    """Generate a model card markdown file."""
    formulation = metadata.get("formulation", "unknown")
    card = f"""# Model Card

## Overview
- **Formulation**: {formulation}
- **Training date**: {metadata.get('training_date', 'unknown')}
- **Training samples**: {metadata.get('n_train_samples', 'unknown')}
- **Features**: {metadata.get('n_features', 'unknown')}
- **Presets**: {args.presets} | Time limit: {args.time_limit}s

## Performance
"""
    if "mean_test_f1" in metrics:
        card += f"- Mean F1: {metrics['mean_test_f1']}\n- Median F1: {metrics['median_test_f1']}\n"
        card += f"- Labels ≥ 0.50: {metrics.get('labels_above_050', 0)}/{metrics.get('labels_total_scored', 0)}\n"
    elif "test_score" in metrics:
        card += f"- {metrics.get('metric', 'score')}: {metrics['test_score']:.4f}\n"
    if "nasa_score" in metrics:
        card += f"- NASA Score: {metrics['nasa_score']:.1f}\n"

    card += f"""
## Intended Use
- Predictive maintenance: {formulation}
- Input: tabular feature vector ({metadata.get('n_features', '?')} features)

## Limitations
- Trained on a single dataset snapshot; performance may degrade with fleet changes
- Requires retraining if new failure modes emerge or sensor configurations change

## Ethical Considerations
- False negatives (missed failures) may cause safety or operational impact
- False positives cause unnecessary maintenance cost
"""
    (args.output / "model_card.md").write_text(card)


def _plot_multilabel_importance(all_importance, metrics, output):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n_plot = min(5, len(all_importance))
    sorted_labels = sorted(all_importance.keys(),
                           key=lambda k: metrics["labels"].get(k, {}).get("test_f1", 0),
                           reverse=True)[:n_plot]
    fig, axes = plt.subplots(n_plot, 1, figsize=(10, 3 * n_plot))
    if n_plot == 1:
        axes = [axes]
    for idx, label in enumerate(sorted_labels):
        feats = all_importance[label][:10]
        if feats:
            axes[idx].barh([f["feature"] for f in feats][::-1], [f["importance"] for f in feats][::-1])
            axes[idx].set_title(label.replace("label_", ""), fontsize=10)
    plt.tight_layout()
    plt.savefig(output / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="AutoGluon PdM training (auto-detects formulation)")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("./model"))
    parser.add_argument("--time-limit", type=int, default=600, help="Seconds (total for single, per-label for multi)")
    parser.add_argument("--presets", default="best", help="AutoGluon preset: medium, good, high, best")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for multi-label training")
    parser.add_argument("--rebalance", action="store_true", help="Apply inverse-frequency sample weights for class imbalance")
    parser.add_argument("--smote", action="store_true", help="Apply SMOTE oversampling for minority class (classification only)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--warm-start", action="store_true", help="Load existing model and refit on new data")
    parser.add_argument("--skip-importance", action="store_true", help="Skip permutation feature importance (saves ~60%% of total time for best_quality)")
    args = parser.parse_args()

    if not args.train.exists():
        sys.exit(f"Error: {args.train} not found")
    if not args.test.exists():
        sys.exit(f"Error: {args.test} not found")

    args.output.mkdir(parents=True, exist_ok=True)

    # Reproducibility: set seeds
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    # Feature selection: drop zero-variance on train, align test
    from pdm.data.utils import drop_zero_variance
    train_df, test_df = drop_zero_variance(train_df, test_df)

    formulation = detect_formulation(train_df)

    print(f"{'='*60}")
    print("  AutoGluon PdM Training")
    print(f"  Formulation: {formulation}")
    print(f"  Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    print(f"{'='*60}")

    # Adaptive budget guidance
    n_rows = len(train_df)
    if n_rows < 1000 and args.time_limit > 300:
        print(f"  ⚠️ Small dataset ({n_rows} rows) with high budget ({args.time_limit}s) — overfitting risk")
    if n_rows > 50000 and args.time_limit < 300:
        print(f"  ⚠️ Large dataset ({n_rows} rows) with low budget ({args.time_limit}s) — may not converge")
    if formulation == "multilabel":
        n_labels = len([c for c in train_df.columns if c.startswith("label_")])
        total_time = n_labels * args.time_limit
        print(f"  ⏱️  Estimated total time: {n_labels} labels × {args.time_limit}s = {total_time//60}m{total_time%60}s")

    if formulation == "multilabel":
        metrics, metadata = train_multilabel(train_df, test_df, args)
    else:
        metrics, metadata = train_single(train_df, test_df, formulation, args)

    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (args.output / "config.json").write_text(json.dumps(
        {"time_limit": args.time_limit, "presets": args.presets}, indent=2
    ))

    # Drift detection baseline: save feature distributions for future monitoring
    feature_cols = metadata.get("feature_names", [])
    if feature_cols:
        baseline = {}
        for col in feature_cols:
            if col in train_df.columns:
                s = train_df[col].dropna()
                baseline[col] = {
                    "mean": float(s.mean()), "std": float(s.std()),
                    "q25": float(s.quantile(0.25)), "q50": float(s.quantile(0.5)),
                    "q75": float(s.quantile(0.75)), "min": float(s.min()), "max": float(s.max()),
                }
        (args.output / "baseline_stats.json").write_text(json.dumps(baseline, indent=2))

    # Save environment fingerprint for reproducibility
    import platform
    env_info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "seed": args.seed,
        "autogluon_version": getattr(__import__("autogluon"), "__version__", "unknown"),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }
    (args.output / "environment.json").write_text(json.dumps(env_info, indent=2))

    # Generate model card
    _generate_model_card(metrics, metadata, args)

    print(f"\n{'='*60}")
    if formulation == "multilabel":
        print(f"  Mean F1: {metrics.get('mean_test_f1', 'N/A')}")
        print(f"  Median F1: {metrics.get('median_test_f1', 'N/A')}")
        print(f"  Labels >= 0.50: {metrics.get('labels_above_050', 0)}/{metrics.get('labels_total_scored', 0)}")
    else:
        print(f"  Test {metrics['metric']} = {metrics['test_score']:.4f}")
    print(f"{'='*60}")
    print(f"✅ Saved to {args.output}/")


if __name__ == "__main__":
    main()
