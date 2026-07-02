#!/usr/bin/env python3
"""Inference script for anomaly detection model (Isolation Forest).

Usage:
    uv run python pdm/anomaly_detection/inference_anomaly.py -n 5
    uv run python pdm/anomaly_detection/inference_anomaly.py --input new_data.csv --model-dir ./anomaly_detection/model
    uv run python pdm/anomaly_detection/inference_anomaly.py --input new_data.csv --top-features 5
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib


def _feature_contributions(model, scaler, sample_scaled, feature_names, top_k=10):
    """Approximate per-feature contribution using mean depth isolation paths."""
    # Use the model's decision_function per-feature perturbation (fast approximation)
    base_score = -model.score_samples(sample_scaled.reshape(1, -1))[0]
    contributions = {}
    for i, fname in enumerate(feature_names):
        perturbed = sample_scaled.copy()
        perturbed[i] = 0.0  # zero out (mean after scaling)
        perturbed_score = -model.score_samples(perturbed.reshape(1, -1))[0]
        contributions[fname] = round(float(base_score - perturbed_score), 6)
    # Sort by absolute contribution
    sorted_contribs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    return sorted_contribs[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Run anomaly detection inference")
    parser.add_argument("-n", type=int, default=5, help="Number of samples to score")
    parser.add_argument("--input", type=Path, default=None, help="Input CSV (default: auto-discover test set)")
    parser.add_argument("--model-dir", type=Path, default=Path("./anomaly_detection/baseline/model"))
    parser.add_argument("--top-features", type=int, default=10, help="Top contributing features to show")
    parser.add_argument("--explain", action="store_true", help="Show per-feature contribution analysis (slower)")
    args = parser.parse_args()

    # Load model artefacts
    metadata = json.loads((args.model_dir / "metadata.json").read_text())
    threshold_data = json.loads((args.model_dir / "threshold.json").read_text())
    model = joblib.load(args.model_dir / "isolation_forest.joblib")
    scaler = joblib.load(args.model_dir / "scaler.joblib")

    feature_names = metadata["feature_names"]
    threshold = threshold_data["threshold"]

    # Auto-discover input
    if args.input is None:
        for candidate in [Path("./data/raw_test.csv"), Path("./data/test.csv")]:
            if candidate.exists():
                args.input = candidate
                break
        if args.input is None:
            parser.error("No input file found. Provide --input explicitly.")

    df = pd.read_csv(args.input, nrows=args.n)
    X = scaler.transform(df[feature_names].fillna(0).values)
    scores = -model.score_samples(X)

    print(f"{'='*60}")
    print("  Anomaly Detection Inference")
    print(f"  Model: {metadata['model_type']} | Features: {len(feature_names)}")
    print(f"  Threshold: {threshold:.6f} | Input: {args.input}")
    print(f"{'='*60}\n")

    # Per-sample latency measurement (score + explain, excludes model loading)
    import time
    latencies_ms = []

    for i in range(len(df)):
        t0 = time.perf_counter()
        score_i = -model.score_samples(X[i:i+1])[0]
        is_anomaly = score_i > threshold
        contribs = _feature_contributions(model, scaler, X[i], feature_names, args.top_features) if args.explain else None
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        flag = "🚨 ANOMALY" if is_anomaly else "✓ Normal"

        # Ground truth (if labels available)
        label_cols = [c for c in df.columns if c.startswith("label_")]
        if label_cols:
            active_labels = [c for c in label_cols if df[c].iloc[i] == 1]
            expected = "ANOMALY" if active_labels else "Normal"
            correct = "✓" if (is_anomaly == bool(active_labels)) else "✗"
        else:
            active_labels = None
            expected = None
            correct = ""

        print(f"{'─'*60}")
        print(f"Sample {i}: score={scores[i]:.6f}  {flag}")
        if expected is not None:
            print(f"  Expected: {expected} {correct}")
            if active_labels:
                print(f"  Active labels: {[c.replace('label_', '') for c in active_labels]}")

        # Top contributors (only with --explain)
        if contribs:
            print(f"  Top contributors (score delta when zeroed):")
            for fname, delta in contribs:
                raw_val = df[fname].iloc[i] if fname in df.columns else 0
                direction = "↑" if delta > 0 else "↓"
                print(f"    {fname:40s} value={raw_val:<10.4f} contribution={delta:+.6f} {direction}")

    print(f"{'─'*60}")
    print(f"\nSummary: {(scores > threshold).sum()}/{len(df)} samples flagged as anomalies")
    print(f"Latency: TP50={np.median(latencies_ms):.1f}ms | TP90={np.percentile(latencies_ms, 90):.1f}ms | TP99={np.percentile(latencies_ms, 99):.1f}ms")


if __name__ == "__main__":
    main()
