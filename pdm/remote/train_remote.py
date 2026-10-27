#!/usr/bin/env python3
"""SageMaker container entry script for pdm-model training.

Runs inside the SageMaker container. Reads data from /opt/ml/input/data/train/,
trains the model based on hyperparameters, saves artifacts to /opt/ml/model/.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    # SageMaker passes hyperparameters as env var
    hps = json.loads(os.environ.get("SM_HPS", "{}"))
    data_dir = Path(os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    output_dir = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))

    formulation = hps.get("formulation", "rul")
    backend = hps.get("backend", "autogluon")
    window_size = int(hps.get("window-size", 30))
    rul_cap = int(hps.get("rul-cap", 125))
    time_limit = int(hps.get("time-limit", 300))
    n_trials = int(hps.get("n-trials", 50))
    optimization_priority = hps.get("optimization-priority", "balanced")

    print(f"Formulation: {formulation}, Backend: {backend}")
    print(f"Window: {window_size}, RUL cap: {rul_cap}, Time: {time_limit}s, Trials: {n_trials}")

    # Load data
    train_df = pd.read_csv(data_dir / "raw_train.csv")
    test_df = pd.read_csv(data_dir / "raw_test.csv")
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")

    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if formulation == "rul":
        from pdm.rul.model import RULPredictor
        model = RULPredictor(window_size=window_size, rul_cap=rul_cap)
        result = model.train(train_df, test_df, time_limit=time_limit, backend=backend,
                             n_trials=n_trials, output=model_dir)
        model.save(model_dir)

    elif formulation == "classification":
        # Use AutoGluon via fault_prediction train logic
        from autogluon.tabular import TabularPredictor
        from pdm.fault_prediction.metrics import (
            classification_eval_metric,
            classification_eval_metric_name,
            classification_quality_gate,
            evaluate_binary_classifier,
            normalize_optimization_priority,
            predictor_decision_threshold,
        )
        target = "machine_failure"
        optimization_priority = normalize_optimization_priority(optimization_priority)
        eval_metric = classification_eval_metric(optimization_priority)
        eval_metric_name = classification_eval_metric_name(optimization_priority)
        predictor = TabularPredictor(
            label=target, eval_metric=eval_metric, problem_type="binary",
            path=str(model_dir / "ag_model"), verbosity=1,
        ).fit(train_data=train_df, time_limit=time_limit, presets="best")
        X_test = test_df.drop(columns=[target])
        preds = predictor.predict(X_test)
        y_true = test_df[target].values
        cls_metrics = evaluate_binary_classifier(predictor, X_test, y_true, y_pred=preds)
        result_metrics = {
            "f1": round(float(cls_metrics["f1"]), 4),
            "precision": round(float(cls_metrics["precision"]), 4),
            "recall": round(float(cls_metrics["recall"]), 4),
            "roc_auc": None if cls_metrics["roc_auc"] is None else round(float(cls_metrics["roc_auc"]), 4),
            "average_precision": None if cls_metrics["average_precision"] is None else round(float(cls_metrics["average_precision"]), 4),
            "optimization_priority": optimization_priority,
            "eval_metric": eval_metric_name,
            "decision_threshold": round(predictor_decision_threshold(predictor), 6),
            "positive_rate": round(float(train_df[target].mean()), 5),
            "quality_gate": classification_quality_gate(
                cls_metrics["f1"], cls_metrics["roc_auc"], cls_metrics["average_precision"],
                float(train_df[target].mean()),
            ),
        }
        result = type("R", (), {"metrics": result_metrics})()

    elif formulation == "survival":
        from pdm.survival.model import SurvivalPredictor
        model = SurvivalPredictor()
        result = model.train(train_df, test_df, time_limit=time_limit)
        model.save(model_dir)

    else:
        sys.exit(f"Unknown formulation: {formulation}")

    # Save metrics
    metrics = result.metrics
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n✅ Metrics: {json.dumps(metrics)}")
    print(f"Model saved to {model_dir}")


if __name__ == "__main__":
    main()
