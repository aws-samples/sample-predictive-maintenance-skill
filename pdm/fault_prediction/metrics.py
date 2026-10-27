"""Shared evaluation helpers for binary fault-classification metrics."""

from __future__ import annotations

from functools import partial
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


# AutoGluon pickles the eval_metric into the saved predictor. The score
# functions must therefore be importable wherever the model is loaded,
# including the standalone serving container, which has sklearn but not pdm.
# A functools.partial over sklearn's fbeta_score pickles as a reference to
# sklearn, so never wrap these in functions defined in this module.
# F2 emphasizes recall while still penalizing false alarms; F0.5 emphasizes
# precision while still rewarding failure coverage.
F2_SCORER = make_scorer(
    name="f2",
    score_func=partial(fbeta_score, beta=2.0, zero_division=0),
    optimum=1.0,
    greater_is_better=True,
    needs_class=True,
)
F05_SCORER = make_scorer(
    name="f0.5",
    score_func=partial(fbeta_score, beta=0.5, zero_division=0),
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


# --- Quality gate -----------------------------------------------------------
#
# F1 at the operating point is a fair gate only when positives are reasonably
# common. Below ~1% positives, F1 >= 0.50 is close to unachievable even for a
# useful ranker, so the gate switches to threshold-independent ranking metrics
# (mirroring the anomaly-detection gate, which uses AUROC >= 0.75). Average
# precision is judged as lift over the positive rate, because AP's random
# baseline equals the positive rate and AUROC alone can look strong at extreme
# imbalance while precision stays near zero.
GATE_MIN_F1 = 0.50
GATE_LOW_PREVALENCE_RATE = 0.01
GATE_MIN_ROC_AUC = 0.75
GATE_MIN_AP_LIFT = 5.0


def classification_quality_gate(
    f1: float | None,
    roc_auc: float | None,
    average_precision: float | None,
    positive_rate: float | None,
) -> dict[str, Any]:
    """Evaluate the imbalance-aware classification quality gate for one label.

    Args:
        f1: Test F1 at the deployed (validation-calibrated) threshold.
        roc_auc: Test ROC AUC, or ``None`` when undefined.
        average_precision: Test average precision, or ``None`` when undefined.
        positive_rate: Positive rate of the training labels before any
            resampling. ``None`` falls back to the F1 gate.

    Returns:
        ``{"passed", "criterion", "positive_rate", "reason"}`` where criterion is
        ``"f1"`` (positive rate >= 1%) or ``"ranking"`` (positive rate < 1%).
    """
    rate = None if positive_rate is None else float(positive_rate)

    if rate is None or rate >= GATE_LOW_PREVALENCE_RATE:
        passed = f1 is not None and float(f1) >= GATE_MIN_F1
        shown = "N/A" if f1 is None else f"{float(f1):.4f}"
        reason = f"F1={shown} {'≥' if passed else '<'} {GATE_MIN_F1:.2f}"
        if rate is not None:
            reason += f" (positive rate {rate:.2%})"
        return {"passed": bool(passed), "criterion": "f1", "positive_rate": rate, "reason": reason}

    prefix = f"positive rate {rate:.2%} < {GATE_LOW_PREVALENCE_RATE:.0%} → ranking gate: "
    if roc_auc is None or average_precision is None:
        return {
            "passed": False, "criterion": "ranking", "positive_rate": rate,
            "reason": prefix + "ROC AUC / AP undefined (test split has a single class)",
        }

    lift = float(average_precision) / rate if rate > 0 else float("inf")
    auc_ok = float(roc_auc) >= GATE_MIN_ROC_AUC
    lift_ok = lift >= GATE_MIN_AP_LIFT
    reason = (
        f"ROC AUC={float(roc_auc):.4f} {'≥' if auc_ok else '<'} {GATE_MIN_ROC_AUC:.2f}, "
        f"AP lift={lift:.1f}× {'≥' if lift_ok else '<'} {GATE_MIN_AP_LIFT:.0f}×"
    )
    return {
        "passed": bool(auc_ok and lift_ok), "criterion": "ranking",
        "positive_rate": rate, "reason": prefix + reason,
    }


def multilabel_quality_gate(label_gates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-label gates: pass when at least half of scored labels pass.

    With every label on the F1 criterion this is the previous "median F1 >= 0.50"
    rule; it additionally lets rare labels be judged by the ranking gate.
    """
    n_scored = len(label_gates)
    n_passed = sum(1 for gate in label_gates.values() if gate.get("passed"))
    passed = n_scored > 0 and 2 * n_passed >= n_scored
    return {
        "passed": bool(passed),
        "labels_passed": n_passed,
        "labels_scored": n_scored,
        "reason": f"{n_passed}/{n_scored} labels pass their gate (need at least half)",
    }


def format_quality_gate(gate: dict[str, Any]) -> str:
    """One-line, human-readable gate result for training logs."""
    return f"Quality gate: {'✅ PASSED' if gate['passed'] else '❌ FAILED'} — {gate['reason']}"
