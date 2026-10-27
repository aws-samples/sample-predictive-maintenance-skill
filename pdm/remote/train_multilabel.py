#!/usr/bin/env python3
"""SageMaker container entry script for multi-label fault prediction training.

Runs inside the SageMaker container. Reads pre-processed train/test CSVs from
/opt/ml/input/data/train/, trains one TabularPredictor per label_* column,
saves metrics + model artifacts to /opt/ml/model/.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    hps = json.loads(os.environ.get("SM_HPS", "{}"))
    data_dir = Path(os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))

    time_limit = int(hps.get("time-limit", 120))
    presets = hps.get("presets", "best")
    experiment_name = hps.get("experiment-name", "unknown")
    workers = int(hps.get("workers", 1))
    optimization_priority = hps.get("optimization-priority", "balanced")

    print(f"Experiment: {experiment_name}, Time/label: {time_limit}s, Presets: {presets}, Workers: {workers}")

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")

    # Drop zero-variance features
    label_cols = sorted([c for c in train_df.columns if c.startswith("label_")])
    feature_cols = [c for c in train_df.columns if c not in label_cols]
    zv = [c for c in feature_cols if train_df[c].nunique() <= 1]
    if zv:
        print(f"Dropping {len(zv)} zero-variance features")
        train_df = train_df.drop(columns=zv)
        test_df = test_df.drop(columns=[c for c in zv if c in test_df.columns])
        feature_cols = [c for c in feature_cols if c not in zv]

    model_dir.mkdir(parents=True, exist_ok=True)

    from autogluon.tabular import TabularPredictor
    from pdm.fault_prediction.metrics import (
        classification_eval_metric,
        classification_eval_metric_name,
        evaluate_binary_classifier,
        normalize_optimization_priority,
        predictor_decision_threshold,
    )

    optimization_priority = normalize_optimization_priority(optimization_priority)
    eval_metric = classification_eval_metric(optimization_priority)
    eval_metric_name = classification_eval_metric_name(optimization_priority)
    metrics = {
        "formulation": "multi-label classification",
        "experiment": experiment_name,
        "optimization_priority": optimization_priority,
        "eval_metric": eval_metric_name,
        "labels": {},
    }

    def _train_one_label(label):
        n_pos = int(train_df[label].sum())
        pos_rate = train_df[label].mean()
        if n_pos < 10:
            return label, {"skipped": True, "reason": "too_few_positives"}

        predictor = TabularPredictor(
            label=label, eval_metric=eval_metric, problem_type="binary",
            path=str(model_dir / "ag_model" / label), verbosity=0,
        ).fit(train_data=train_df[feature_cols + [label]], time_limit=time_limit, presets=presets)

        X_test = test_df[feature_cols]
        y_pred = predictor.predict(X_test)
        y_true = test_df[label].values
        cls_metrics = evaluate_binary_classifier(predictor, X_test, y_true, y_pred=y_pred)

        top_feats = []
        try:
            importance = predictor.feature_importance(test_df[feature_cols + [label]], silent=True)
            top_feats = list(importance.head(5).index)
        except Exception:
            pass

        return label, {
            "test_f1": round(cls_metrics["f1"], 4), "test_precision": round(cls_metrics["precision"], 4),
            "test_recall": round(cls_metrics["recall"], 4),
            "test_roc_auc": None if cls_metrics["roc_auc"] is None else round(cls_metrics["roc_auc"], 4),
            "test_average_precision": None if cls_metrics["average_precision"] is None else round(cls_metrics["average_precision"], 4),
            "positive_rate": round(pos_rate, 5),
            "optimization_priority": optimization_priority,
            "eval_metric": eval_metric_name,
            "decision_threshold": round(predictor_decision_threshold(predictor), 6),
            "best_model": predictor.model_best, "top_features": top_feats,
        }

    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_train_one_label, lbl): lbl for lbl in label_cols}
            for i, future in enumerate(as_completed(futures)):
                label, result = future.result()
                metrics["labels"][label] = result
                status = f"F1={result['test_f1']} ROC-AUC={result.get('test_roc_auc')} AP={result.get('test_average_precision')}" if "test_f1" in result else "SKIPPED"
                print(f"  [{i+1}/{len(label_cols)}] {label}: {status}")
    else:
        for i, label in enumerate(label_cols):
            print(f"\n[{i+1}/{len(label_cols)}] Training {label}...")
            label, result = _train_one_label(label)
            metrics["labels"][label] = result
            if "test_f1" in result:
                print(f"  {label}: F1={result['test_f1']:.4f} P={result['test_precision']:.4f} R={result['test_recall']:.4f} ROC-AUC={result.get('test_roc_auc')} AP={result.get('test_average_precision')}")

    scored = {k: v for k, v in metrics["labels"].items() if "test_f1" in v}
    if scored:
        f1s = [v["test_f1"] for v in scored.values()]
        metrics["mean_test_f1"] = round(float(np.mean(f1s)), 4)
        metrics["median_test_f1"] = round(float(np.median(f1s)), 4)
        roc_aucs = [v["test_roc_auc"] for v in scored.values() if v.get("test_roc_auc") is not None]
        avg_precisions = [v["test_average_precision"] for v in scored.values() if v.get("test_average_precision") is not None]
        if roc_aucs:
            metrics["mean_test_roc_auc"] = round(float(np.mean(roc_aucs)), 4)
            metrics["median_test_roc_auc"] = round(float(np.median(roc_aucs)), 4)
        if avg_precisions:
            metrics["mean_test_average_precision"] = round(float(np.mean(avg_precisions)), 4)
            metrics["median_test_average_precision"] = round(float(np.median(avg_precisions)), 4)

    metadata = {
        "feature_names": feature_cols, "label_names": label_cols,
        "formulation": "multi-label classification",
        "experiment": experiment_name,
        "n_train_samples": len(train_df), "n_features": len(feature_cols),
    }

    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    # Keep the legacy summary line stable for pdm.remote.parallel log parsing.
    print(f"\n✅ Metrics: mean_f1={metrics.get('mean_test_f1')}, median_f1={metrics.get('median_test_f1')}")
    print(f"Ranking metrics: mean_roc_auc={metrics.get('mean_test_roc_auc')}, mean_average_precision={metrics.get('mean_test_average_precision')}")


if __name__ == "__main__":
    main()
