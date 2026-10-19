"""Tests for threshold-independent binary classification metrics."""

import numpy as np
import pandas as pd
import pytest

from pdm.fault_prediction.metrics import (
    compute_binary_metrics,
    evaluate_binary_classifier,
    positive_class_scores,
)


class _FakeBinaryPredictor:
    def __init__(self, scores, predictions, positive_class="failure"):
        self._scores = scores
        self._predictions = predictions
        self.positive_class = positive_class
        self.as_multiclass = None

    def predict_proba(self, features, as_multiclass=True):
        self.as_multiclass = as_multiclass
        return pd.Series(self._scores, index=features.index)

    def predict(self, features):
        return pd.Series(self._predictions, index=features.index)


def test_ranking_metrics_separate_ranking_from_operating_threshold():
    X = pd.DataFrame({"sensor": [0.1, 0.2, 0.3, 0.4]})
    y_true = np.array(["normal", "normal", "failure", "failure"])
    predictor = _FakeBinaryPredictor(
        scores=[0.10, 0.20, 0.80, 0.90],
        predictions=["normal", "normal", "normal", "normal"],
    )

    metrics = evaluate_binary_classifier(predictor, X, y_true)

    assert predictor.as_multiclass is False
    assert metrics["f1"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)


def test_positive_class_scores_do_not_depend_on_probability_column_order():
    X = pd.DataFrame({"sensor": [1.0, 2.0, 3.0]})
    predictor = _FakeBinaryPredictor(
        scores=[0.25, 0.50, 0.75],
        predictions=["normal", "failure", "failure"],
        positive_class="failure",
    )

    scores, positive_class = positive_class_scores(predictor, X)

    np.testing.assert_allclose(scores, [0.25, 0.50, 0.75])
    assert positive_class == "failure"
    assert predictor.as_multiclass is False


def test_one_class_evaluation_returns_null_ranking_metrics():
    metrics = compute_binary_metrics(
        y_true_binary=[0, 0, 0],
        y_pred_binary=[0, 0, 0],
        y_score=[0.1, 0.2, 0.3],
    )

    assert metrics["f1"] == 0.0
    assert metrics["roc_auc"] is None
    assert metrics["average_precision"] is None


def test_non_finite_probability_is_rejected():
    X = pd.DataFrame({"sensor": [1.0, 2.0]})
    predictor = _FakeBinaryPredictor(
        scores=[0.2, np.nan],
        predictions=["normal", "failure"],
    )

    with pytest.raises(ValueError, match="NaN or infinite"):
        positive_class_scores(predictor, X)


def test_numeric_positive_class_is_supported():
    X = pd.DataFrame({"sensor": [1.0, 2.0, 3.0, 4.0]})
    predictor = _FakeBinaryPredictor(
        scores=[0.05, 0.80, 0.10, 0.90],
        predictions=[0, 1, 0, 1],
        positive_class=1,
    )

    metrics = evaluate_binary_classifier(predictor, X, np.array([0, 1, 0, 1]))

    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)
