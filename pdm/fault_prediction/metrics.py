"""Shared evaluation helpers for binary fault-classification metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from autogluon.core.metrics import make_scorer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _f2_score(y_true, y_pred) -> float:
    """F2: emphasize recall while still penalizing false alarms."""
    return float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))


def _f05_score(y_true, y_pred) -> float:
    """F0.5: emphasize precision while still rewarding failure coverage."""
    return float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0))


F2_SCORER = make_scorer(
    name="f2",
    score_func=_f2_score,
    optimum=1.0,
    greater_is_better=True,
    needs_class=True,
)
F05_SCORER = make_scorer(
    name="f0.5",
    score_func=_f05_score,
    optimum=1.0,
    greater_is_better=True,
    needs_class=True,
)

OPTIMIZATION_METRICS = {
    "balanced": "f1",
    "recall": F2_SCORER,
    "precision": F05_SCORER,
}
OPTIMIZATION_METRIC_NAMES = {
    "balanced": "f1",
    "recall": "f2",
    "precision": "f0.5",
}


def normalize_optimization_priority(priority: str = "balanced") -> str:
    """Normalize the user-facing classification optimization priority."""
    normalized = str(priority).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {"f1": "balanced", "balanced_f1": "balanced"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in OPTIMIZATION_METRICS:
        choices = ", ".join(OPTIMIZATION_METRICS)
        raise ValueError(f"Unknown optimization priority {priority!r}; choose one of: {choices}")
    return normalized


def classification_eval_metric(priority: str = "balanced"):
    """Return the AutoGluon scorer that matches the operating priority.

    Recall and precision priorities use F2 and F0.5 rather than pure recall or
    precision. This preserves the requested bias without selecting degenerate
    all-positive or almost-all-negative operating points.
    """
    return OPTIMIZATION_METRICS[normalize_optimization_priority(priority)]


def classification_eval_metric_name(priority: str = "balanced") -> str:
    """Return the stable, JSON-safe name of the priority's evaluation metric."""
    return OPTIMIZATION_METRIC_NAMES[normalize_optimization_priority(priority)]


def predictor_decision_threshold(predictor) -> float:
    """Return a validated binary decision threshold, defaulting to 0.5."""
    threshold = float(getattr(predictor, "decision_threshold", 0.5))
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Invalid predictor decision threshold: {threshold}")
    return threshold


def binarize_labels(values, positive_class: Any) -> np.ndarray:
    """Map arbitrary binary labels to 0/1 using AutoGluon's positive class."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return (arr == positive_class).astype(int)


def positive_class_scores(predictor, features: pd.DataFrame) -> tuple[np.ndarray, Any]:
    """Return P(positive_class) safely for an AutoGluon binary predictor.

    ``as_multiclass=False`` avoids relying on probability-column ordering and
    AutoGluon defines the returned series as the probability of
    ``predictor.positive_class``.
    """
    positive_class = getattr(predictor, "positive_class", None)
    if positive_class is None:
        raise ValueError("Binary predictor does not expose positive_class")

    raw_scores = predictor.predict_proba(features, as_multiclass=False)
    scores = np.asarray(raw_scores, dtype=float).reshape(-1)
    if len(scores) != len(features):
        raise ValueError(
            f"Probability length mismatch: got {len(scores)} scores for {len(features)} rows"
        )
    if not np.isfinite(scores).all():
        raise ValueError("Predicted probabilities contain NaN or infinite values")
    return scores, positive_class


def compute_binary_metrics(
    y_true_binary,
    y_pred_binary,
    y_score=None,
) -> dict[str, float | None]:
    """Compute operating-point and ranking metrics from 0/1 arrays.

    ROC AUC and average precision require both classes in the evaluation split.
    When the split contains only one class, both ranking metrics are returned as
    ``None`` so JSON serialization produces ``null`` rather than a misleading
    value or NaN.
    """
    y_true = np.asarray(y_true_binary, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred_binary, dtype=int).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("y_true_binary must contain only 0/1 values")
    if not set(np.unique(y_pred)).issubset({0, 1}):
        raise ValueError("y_pred_binary must contain only 0/1 values")

    metrics: dict[str, float | None] = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None,
        "average_precision": None,
    }

    if y_score is None:
        return metrics

    scores = np.asarray(y_score, dtype=float).reshape(-1)
    if len(scores) != len(y_true):
        raise ValueError("y_score and y_true must have the same length")
    if not np.isfinite(scores).all():
        raise ValueError("Predicted probabilities contain NaN or infinite values")

    if np.unique(y_true).size < 2:
        return metrics

    metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
    metrics["average_precision"] = float(average_precision_score(y_true, scores))
    return metrics


def evaluate_binary_classifier(
    predictor,
    features: pd.DataFrame,
    y_true,
    y_pred=None,
) -> dict[str, float | None]:
    """Evaluate an AutoGluon binary predictor without assuming class order."""
    scores, positive_class = positive_class_scores(predictor, features)
    predictions = predictor.predict(features) if y_pred is None else y_pred
    return compute_binary_metrics(
        binarize_labels(y_true, positive_class),
        binarize_labels(predictions, positive_class),
        scores,
    )
