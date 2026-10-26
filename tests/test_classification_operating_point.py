"""Regression coverage for validation-only classification operating points."""

import pandas as pd
import pytest

from pdm.fault_prediction import model as model_module
from pdm.fault_prediction.metrics import (
    classification_eval_metric,
    classification_eval_metric_name,
    normalize_optimization_priority,
)


class _CalibratingPredictor:
    positive_class = 1
    model_best = "FakeModel"
    created = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.decision_threshold = 0.5
        self.fit_calibration = None
        type(self).created.append(self)

    def fit(self, train_data, time_limit=None, presets=None, calibrate_decision_threshold="auto"):
        self.fit_calibration = calibrate_decision_threshold
        metric_name = getattr(self.kwargs["eval_metric"], "name", self.kwargs["eval_metric"])
        self.decision_threshold = {"f1": 0.41, "f2": 0.23, "f0.5": 0.79}[metric_name]
        return self

    def predict_proba(self, features, as_multiclass=True):
        assert as_multiclass is False
        return pd.Series(features["score"].astype(float).values, index=features.index)

    def predict(self, features):
        return (features["score"].astype(float) > self.decision_threshold).astype(int)


def _frames():
    train = pd.DataFrame({
        "score": [0.05, 0.10, 0.15, 0.20, 0.80, 0.85, 0.90, 0.95, 0.12, 0.88] * 2,
        "machine_failure": [0, 0, 0, 0, 1, 1, 1, 1, 0, 1] * 2,
    })
    test = pd.DataFrame({
        "score": [0.10, 0.30, 0.70, 0.90],
        "machine_failure": [0, 0, 1, 1],
    })
    return train, test


@pytest.mark.parametrize(
    ("priority", "normalized", "metric"),
    [
        ("balanced", "balanced", "f1"),
        ("f1", "balanced", "f1"),
        ("balanced-f1", "balanced", "f1"),
        ("recall", "recall", "f2"),
        ("precision", "precision", "f0.5"),
    ],
)
def test_priority_mapping(priority, normalized, metric):
    assert normalize_optimization_priority(priority) == normalized
    assert classification_eval_metric_name(priority) == metric
    scorer = classification_eval_metric(priority)
    assert getattr(scorer, "name", scorer) == metric


def test_unknown_priority_is_rejected():
    with pytest.raises(ValueError, match="Unknown optimization priority"):
        classification_eval_metric("accuracy")


def test_priority_scorers_bias_without_degenerate_objectives():
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    precision_heavy = [1, 1, 0, 0, 0, 0, 0, 0]
    recall_heavy = [1, 1, 1, 1, 1, 1, 0, 0]
    assert classification_eval_metric("recall")(y_true, recall_heavy) > classification_eval_metric("recall")(y_true, precision_heavy)
    assert classification_eval_metric("precision")(y_true, precision_heavy) > classification_eval_metric("precision")(y_true, recall_heavy)


def test_failure_classifier_uses_validation_calibrated_threshold(monkeypatch, tmp_path):
    _CalibratingPredictor.created.clear()
    monkeypatch.setattr(model_module, "TabularPredictor", _CalibratingPredictor)
    train, test = _frames()

    model = model_module.FailureClassifier()
    result = model.train(
        train,
        test,
        time_limit=1,
        output=tmp_path,
        optimization_priority="recall",
    )

    predictor = _CalibratingPredictor.created[-1]
    metrics = result.metrics["per_label"]["machine_failure"]
    assert getattr(predictor.kwargs["eval_metric"], "name", predictor.kwargs["eval_metric"]) == "f2"
    assert predictor.fit_calibration == "auto"
    assert model.thresholds["machine_failure"] == pytest.approx(0.23)
    assert metrics["decision_threshold"] == pytest.approx(0.23)
    assert metrics["optimization_priority"] == "recall"


def test_legacy_threshold_switch_never_tunes_on_test_labels(monkeypatch, tmp_path):
    _CalibratingPredictor.created.clear()
    monkeypatch.setattr(model_module, "TabularPredictor", _CalibratingPredictor)
    train, test = _frames()
    flipped = test.copy()
    flipped["machine_failure"] = 1 - flipped["machine_failure"]

    first = model_module.FailureClassifier()
    first.train(
        train, test, time_limit=1, output=tmp_path / "first",
        optimization_priority="precision", optimize_threshold=True,
    )
    second = model_module.FailureClassifier()
    second.train(
        train, flipped, time_limit=1, output=tmp_path / "second",
        optimization_priority="precision", optimize_threshold=True,
    )

    assert first.thresholds["machine_failure"] == pytest.approx(0.79)
    assert second.thresholds["machine_failure"] == pytest.approx(0.79)
    assert _CalibratingPredictor.created[-2].fit_calibration is True
    assert _CalibratingPredictor.created[-1].fit_calibration is True


def test_predict_uses_stored_operating_threshold(monkeypatch, tmp_path):
    _CalibratingPredictor.created.clear()
    monkeypatch.setattr(model_module, "TabularPredictor", _CalibratingPredictor)
    train, test = _frames()
    model = model_module.FailureClassifier()
    model.train(
        train, test, time_limit=1, output=tmp_path, optimization_priority="recall"
    )

    predictions = model.predict(pd.DataFrame({"score": [0.20, 0.24]})).predictions
    assert predictions["machine_failure_pred"].tolist() == [0, 1]
