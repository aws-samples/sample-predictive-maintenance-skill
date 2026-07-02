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
        target = "machine_failure"
        predictor = TabularPredictor(
            label=target, eval_metric="f1", problem_type="binary",
            path=str(model_dir / "ag_model"), verbosity=1,
        ).fit(train_data=train_df, time_limit=time_limit, presets="best")
        from sklearn.metrics import f1_score, precision_score, recall_score
        preds = predictor.predict(test_df.drop(columns=[target]))
        y_true = test_df[target].values
        result_metrics = {
            "f1": round(float(f1_score(y_true, preds)), 4),
            "precision": round(float(precision_score(y_true, preds)), 4),
            "recall": round(float(recall_score(y_true, preds)), 4),
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
