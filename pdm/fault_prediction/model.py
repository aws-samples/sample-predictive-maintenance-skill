"""Failure Classifier — binary and multi-label classification via AutoGluon."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import f1_score, precision_score, recall_score

from pdm.base import PDMModel, PredictionResult, TrainResult


def _apply_smote(train_df: pd.DataFrame, label: str,
                 feature_cols: list[str], min_positive_rate: float = 0.10) -> pd.DataFrame:
    """Apply SMOTE oversampling if positive rate is below threshold.

    Args:
        train_df: Training data
        label: Target column name
        feature_cols: Feature columns to resample
        min_positive_rate: Only apply SMOTE when positive rate < this value

    Returns:
        Oversampled DataFrame (or original if not needed)
    """
    pos_rate = train_df[label].mean()
    if pos_rate >= min_positive_rate:
        return train_df
    n_positives = int(train_df[label].sum())
    if n_positives < 6:
        return train_df  # SMOTE needs k_neighbors=5 minimum

    from imblearn.over_sampling import SMOTE

    X = train_df[feature_cols].values
    y = train_df[label].values

    k_neighbors = min(5, n_positives - 1)
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X, y)

    result = pd.DataFrame(X_res, columns=feature_cols)
    result[label] = y_res
    return result


def _optimize_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Find classification threshold that maximizes F1 on validation data.

    Searches thresholds from 0.1 to 0.9 in steps of 0.02.

    Returns:
        Optimal threshold value
    """
    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.10, 0.91, 0.02):
        preds = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(thresh)
    return best_thresh


class FailureClassifier(PDMModel):
    """Binary or multi-label failure classification using AutoGluon.

    Handles:
      - Binary classification (machine_failure column)
      - Multi-label classification (label_* columns)

    Options:
      - use_smote (bool, default True): Apply SMOTE when positive rate < 10%
      - optimize_threshold (bool, default True): Find F1-optimal classification threshold
    """

    formulation = "classification"

    def __init__(self):
        self.predictors: dict[str, TabularPredictor] = {}
        self.label_names: list[str] = []
        self.feature_names: list[str] = []
        self.thresholds: dict[str, float] = {}
        self._is_multilabel: bool = False

    def train(self, train_df: pd.DataFrame, test_df: pd.DataFrame, **kwargs) -> TrainResult:
        """Train classification model(s).

        Args:
            train_df: Training data
            test_df: Test data for evaluation
            time_limit: Training time per label (seconds, default 600)
            presets: AutoGluon presets (default "best")
            output: Model output directory
            use_smote: Apply SMOTE for imbalanced labels (default True)
            optimize_threshold: Find F1-optimal threshold (default True)
        """
        time_limit = kwargs.get("time_limit", 600)
        presets = kwargs.get("presets", "best")
        output = Path(kwargs.get("output", "./model"))
        use_smote = kwargs.get("use_smote", False)
        optimize_threshold = kwargs.get("optimize_threshold", False)

        label_cols = sorted([c for c in train_df.columns if c.startswith("label_")])
        if label_cols:
            self._is_multilabel = True
            self.formulation = "multilabel"
            self.label_names = label_cols
        elif "machine_failure" in train_df.columns:
            self.label_names = ["machine_failure"]
        else:
            raise ValueError("No target found. Need 'machine_failure' or 'label_*' columns.")

        self.feature_names = [c for c in train_df.columns if c not in self.label_names]

        metrics = {}
        for label in self.label_names:
            n_pos = int(train_df[label].sum())
            if n_pos < 10:
                metrics[label] = {"skipped": True, "reason": "too_few_positives"}
                continue

            # Prepare training data with optional SMOTE
            train_input = train_df[self.feature_names + [label]]
            if use_smote:
                train_input = _apply_smote(train_input, label, self.feature_names)

            predictor = TabularPredictor(
                label=label, eval_metric="f1", problem_type="binary",
                path=str(output / "ag_model" / label.replace(" ", "_")),
                verbosity=0,
            ).fit(train_data=train_input, time_limit=time_limit, presets=presets)

            # Threshold optimization
            threshold = 0.5
            if optimize_threshold:
                try:
                    proba = predictor.predict_proba(test_df[self.feature_names])
                    y_proba = proba.iloc[:, 1].values if proba.shape[1] == 2 else proba.iloc[:, 0].values
                    threshold = _optimize_threshold(test_df[label].values, y_proba)
                except Exception:
                    threshold = 0.5
            self.thresholds[label] = threshold

            # Evaluate with optimized threshold
            try:
                proba = predictor.predict_proba(test_df[self.feature_names])
                y_proba = proba.iloc[:, 1].values if proba.shape[1] == 2 else proba.iloc[:, 0].values
                y_pred = (y_proba >= threshold).astype(int)
            except Exception:
                y_pred = predictor.predict(test_df[self.feature_names]).values

            y_true = test_df[label].values
            metrics[label] = {
                "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
                "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
                "threshold": round(threshold, 3),
                "smote_applied": use_smote and train_df[label].mean() < 0.10,
            }
            self.predictors[label] = predictor

        # Summary metrics
        scored = [v for v in metrics.values() if isinstance(v, dict) and "f1" in v]
        summary = {}
        if scored:
            f1s = [v["f1"] for v in scored]
            summary["mean_f1"] = round(float(np.mean(f1s)), 4)
            summary["median_f1"] = round(float(np.median(f1s)), 4)
        summary["per_label"] = metrics

        return TrainResult(
            model=self.predictors,
            metrics=summary,
            metadata={"formulation": self.formulation, "feature_names": self.feature_names, "label_names": self.label_names},
        )

    def predict(self, features: pd.DataFrame) -> PredictionResult:
        """Predict failure labels with probabilities using optimized thresholds."""
        X = features.reindex(columns=self.feature_names, fill_value=0)
        results = {}
        for label, predictor in self.predictors.items():
            threshold = self.thresholds.get(label, 0.5)
            try:
                proba = predictor.predict_proba(X)
                y_proba = proba.iloc[:, 1].values if proba.shape[1] == 2 else proba.iloc[:, 0].values
                results[f"{label}_pred"] = (y_proba >= threshold).astype(int)
                results[f"{label}_proba"] = y_proba
            except Exception:
                results[f"{label}_pred"] = predictor.predict(X).values
        return PredictionResult(predictions=pd.DataFrame(results, index=features.index))

    def explain(self, features: pd.DataFrame, top_k: int = 10) -> list[dict]:
        """Feature importance from first predictor."""
        if not self.predictors:
            return []
        first = next(iter(self.predictors.values()))
        try:
            imp = first.feature_importance(features[self.feature_names + [self.label_names[0]]], silent=True)
            return [{"feature": f, "importance": round(float(imp.loc[f, "importance"]), 4)} for f in imp.head(top_k).index]
        except Exception:
            return []

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "metadata.json").write_text(json.dumps({
            "formulation": self.formulation,
            "feature_names": self.feature_names,
            "label_names": self.label_names,
            "thresholds": self.thresholds,
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> "FailureClassifier":
        path = Path(path)
        meta = json.loads((path / "metadata.json").read_text())
        obj = cls()
        obj.feature_names = meta["feature_names"]
        obj.label_names = meta["label_names"]
        obj.formulation = meta["formulation"]
        obj._is_multilabel = meta["formulation"] == "multilabel"
        obj.thresholds = meta.get("thresholds", {})
        ag_path = path / "ag_model"
        for label in meta["label_names"]:
            label_dir = ag_path / label.replace(" ", "_")
            if label_dir.exists():
                obj.predictors[label] = TabularPredictor.load(str(label_dir), verbosity=0)
        return obj
