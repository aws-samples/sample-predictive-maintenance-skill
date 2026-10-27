"""Failure Classifier — binary and multi-label classification via AutoGluon."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from pdm.base import PDMModel, PredictionResult, TrainResult
from pdm.fault_prediction.metrics import (
    binarize_labels,
    classification_eval_metric,
    classification_eval_metric_name,
    classification_quality_gate,
    compute_binary_metrics,
    evaluate_binary_classifier,
    multilabel_quality_gate,
    normalize_optimization_priority,
    positive_class_scores,
    predictor_decision_threshold,
)


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


class FailureClassifier(PDMModel):
    """Binary or multi-label failure classification using AutoGluon.

    Handles:
      - Binary classification (machine_failure column)
      - Multi-label classification (label_* columns)

    Options:
      - use_smote (bool, default False): Apply SMOTE when positive rate < 10%.
        Note: AutoGluon calibrates the decision threshold on validation rows
        drawn from the (oversampled) training input, so with SMOTE enabled the
        threshold is partly tuned on synthetic positives.
      - optimization_priority (str, default "balanced"): balanced, recall, or precision
      - optimize_threshold (bool, optional): Backward-compatible control for AutoGluon's
        validation-only decision-threshold calibration. Test labels are never used.
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
            use_smote: Apply SMOTE for imbalanced labels (default False)
            optimization_priority: balanced, recall, or precision (default "balanced")
            optimize_threshold: Optional legacy switch. If provided, controls AutoGluon's
                validation-only calibration; it never tunes on ``test_df``.
        """
        time_limit = kwargs.get("time_limit", 600)
        presets = kwargs.get("presets", "best")
        output = Path(kwargs.get("output", "./model"))
        use_smote = kwargs.get("use_smote", False)
        optimization_priority = normalize_optimization_priority(
            kwargs.get("optimization_priority", "balanced")
        )
        eval_metric = classification_eval_metric(optimization_priority)
        eval_metric_name = classification_eval_metric_name(optimization_priority)
        legacy_optimize_threshold = kwargs.get("optimize_threshold") if "optimize_threshold" in kwargs else None
        calibrate_decision_threshold = (
            "auto" if legacy_optimize_threshold is None else bool(legacy_optimize_threshold)
        )

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
                label=label, eval_metric=eval_metric, problem_type="binary",
                path=str(output / "ag_model" / label.replace(" ", "_")),
                verbosity=0,
            ).fit(
                train_data=train_input,
                time_limit=time_limit,
                presets=presets,
                calibrate_decision_threshold=calibrate_decision_threshold,
            )

            # AutoGluon calibrates this threshold on its validation data. Test labels
            # are evaluation-only and must never influence the deployable threshold.
            threshold = predictor_decision_threshold(predictor)
            self.thresholds[label] = threshold

            y_true = test_df[label].values
            try:
                y_pred = predictor.predict(test_df[self.feature_names])
                cls_metrics = evaluate_binary_classifier(
                    predictor, test_df[self.feature_names], y_true, y_pred=y_pred
                )
            except Exception:
                y_pred = predictor.predict(test_df[self.feature_names]).values
                positive_class = getattr(predictor, "positive_class", 1)
                cls_metrics = compute_binary_metrics(
                    binarize_labels(y_true, positive_class),
                    binarize_labels(y_pred, positive_class),
                )

            metrics[label] = {
                "f1": round(float(cls_metrics["f1"]), 4),
                "precision": round(float(cls_metrics["precision"]), 4),
                "recall": round(float(cls_metrics["recall"]), 4),
                "roc_auc": None if cls_metrics["roc_auc"] is None else round(float(cls_metrics["roc_auc"]), 4),
                "average_precision": None if cls_metrics["average_precision"] is None else round(float(cls_metrics["average_precision"]), 4),
                "threshold": round(threshold, 6),
                "decision_threshold": round(threshold, 6),
                "optimization_priority": optimization_priority,
                "eval_metric": eval_metric_name,
                "smote_applied": use_smote and train_df[label].mean() < 0.10,
            }
            # Gate on the training positive rate before SMOTE resampling.
            metrics[label]["quality_gate"] = classification_quality_gate(
                cls_metrics["f1"], cls_metrics["roc_auc"], cls_metrics["average_precision"],
                float(train_df[label].mean()),
            )
            self.predictors[label] = predictor

        # Summary metrics
        scored = [v for v in metrics.values() if isinstance(v, dict) and "f1" in v]
        summary = {}
        if scored:
            f1s = [v["f1"] for v in scored]
            summary["mean_f1"] = round(float(np.mean(f1s)), 4)
            summary["median_f1"] = round(float(np.median(f1s)), 4)
            roc_aucs = [v["roc_auc"] for v in scored if v.get("roc_auc") is not None]
            avg_precisions = [v["average_precision"] for v in scored if v.get("average_precision") is not None]
            if roc_aucs:
                summary["mean_roc_auc"] = round(float(np.mean(roc_aucs)), 4)
                summary["median_roc_auc"] = round(float(np.median(roc_aucs)), 4)
            if avg_precisions:
                summary["mean_average_precision"] = round(float(np.mean(avg_precisions)), 4)
                summary["median_average_precision"] = round(float(np.median(avg_precisions)), 4)
            summary["quality_gate"] = multilabel_quality_gate(
                {label: m["quality_gate"] for label, m in metrics.items() if "quality_gate" in m}
            )
        summary["per_label"] = metrics

        return TrainResult(
            model=self.predictors,
            metrics=summary,
            metadata={
                "formulation": self.formulation,
                "feature_names": self.feature_names,
                "label_names": self.label_names,
                "optimization_priority": optimization_priority,
                "eval_metric": eval_metric_name,
            },
        )

    def predict(self, features: pd.DataFrame) -> PredictionResult:
        """Predict failure labels with probabilities using the calibrated thresholds."""
        X = features.reindex(columns=self.feature_names, fill_value=0)
        results = {}
        for label, predictor in self.predictors.items():
            threshold = self.thresholds.get(label, predictor_decision_threshold(predictor))
            try:
                y_proba, _ = positive_class_scores(predictor, X)
                # Match AutoGluon's binary conversion rule: positive when p > threshold.
                results[f"{label}_pred"] = (y_proba > threshold).astype(int)
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
                predictor = TabularPredictor.load(str(label_dir), verbosity=0)
                obj.predictors[label] = predictor
                if label not in obj.thresholds:
                    obj.thresholds[label] = predictor_decision_threshold(predictor)
        return obj
