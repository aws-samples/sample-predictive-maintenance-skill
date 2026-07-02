#!/usr/bin/env python3
"""Evaluate anomaly detection model: metrics, score distribution plot, synthetic injection test.

Usage:
    uv run python pdm/anomaly_detection/evaluate_anomaly.py --model-dir ./anomaly_detection/model --test ./data/raw_test.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pdm.anomaly_detection.synthetic_anomalies import inject_anomalies


def evaluate(args):
    model_dir = args.model_dir
    metadata = json.loads((model_dir / "metadata.json").read_text())
    threshold_data = json.loads((model_dir / "threshold.json").read_text())

    model = joblib.load(model_dir / "isolation_forest.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")
    threshold = threshold_data["threshold"]
    feature_cols = metadata["feature_names"]

    test_df = pd.read_csv(args.test)
    X_test = scaler.transform(test_df[feature_cols].fillna(0))
    test_scores = -model.score_samples(X_test)

    metrics = {
        "threshold": threshold,
        "test_anomaly_rate": float((test_scores > threshold).mean()),
        "test_mean_score": float(test_scores.mean()),
        "test_std_score": float(test_scores.std()),
    }

    # --- Semi-supervised metrics (if labels available) ---
    label_cols = [c for c in test_df.columns if c.startswith("label_")]
    has_labels = False
    if label_cols:
        y_true = (test_df[label_cols].sum(axis=1) > 0).astype(int).values
        has_labels = True
    elif "machine_failure" in test_df.columns:
        y_true = test_df["machine_failure"].values
        has_labels = True

    if has_labels:
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
        y_pred = (test_scores > threshold).astype(int)
        metrics["supervised"] = {
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "auroc": round(float(roc_auc_score(y_true, test_scores)), 4) if y_true.sum() > 0 else None,
            "auprc": round(float(average_precision_score(y_true, test_scores)), 4) if y_true.sum() > 0 else None,
            "n_true_anomalies": int(y_true.sum()),
        }
        print(f"  Supervised metrics (labels available):")
        for k, v in metrics["supervised"].items():
            print(f"    {k}: {v}")

    # --- Synthetic injection test ---
    print("\n  Synthetic injection test (10% injected anomalies):")
    aug_df, syn_labels = inject_anomalies(test_df, feature_cols, fraction=0.10)
    X_aug = scaler.transform(aug_df[feature_cols].fillna(0))
    aug_scores = -model.score_samples(X_aug)
    aug_pred = (aug_scores > threshold).astype(int)

    syn_true = syn_labels.values
    from sklearn.metrics import f1_score, precision_score, recall_score
    syn_detection_rate = float(aug_pred[syn_true == 1].mean())
    metrics["synthetic"] = {
        "detection_rate": round(syn_detection_rate, 4),
        "f1": round(float(f1_score(syn_true, aug_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(syn_true, aug_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(syn_true, aug_pred, zero_division=0)), 4),
    }
    print(f"    Detection rate: {syn_detection_rate:.2%}")
    print(f"    F1: {metrics['synthetic']['f1']}")

    # --- Quality gate ---
    if has_labels:
        passed = metrics["supervised"]["auroc"] is not None and metrics["supervised"]["auroc"] >= 0.75
        gate_reason = f"AUROC={metrics['supervised']['auroc']} {'≥' if passed else '<'} 0.75"
    else:
        passed = syn_detection_rate >= 0.80
        gate_reason = f"Synthetic detection={syn_detection_rate:.2%} {'≥' if passed else '<'} 80%"

    metrics["quality_gate"] = {"passed": passed, "reason": gate_reason}
    print(f"\n  Quality gate: {'✅ PASSED' if passed else '❌ FAILED'} — {gate_reason}")

    # --- Score distribution plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(test_scores, bins=50, alpha=0.7, label="Test scores", color="steelblue", density=True)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold={threshold:.4f}")
    if has_labels and y_true.sum() > 0:
        ax.hist(test_scores[y_true == 1], bins=30, alpha=0.5, label="True anomalies", color="crimson", density=True)
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title("Anomaly Score Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(model_dir / "score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save metrics
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\n✅ Evaluation complete. Results in {model_dir}/")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection model")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    args = parser.parse_args()

    for p in (args.model_dir / "isolation_forest.joblib", args.test):
        if not p.exists():
            sys.exit(f"Error: {p} not found")

    evaluate(args)


if __name__ == "__main__":
    main()
