"""Imbalance-aware classification quality gate (issue #3, R3)."""

from types import SimpleNamespace

import pandas as pd
import pytest

from pdm.fault_prediction import train as training
from pdm.fault_prediction.metrics import (
    classification_quality_gate,
    format_quality_gate,
    multilabel_quality_gate,
)


@pytest.mark.parametrize(
    ("f1", "positive_rate", "passed"),
    [(0.50, 0.05, True), (0.49, 0.05, False), (0.50, 0.01, True), (0.60, None, True)],
)
def test_common_positives_use_f1_gate(f1, positive_rate, passed):
    gate = classification_quality_gate(f1, 0.99, 0.99, positive_rate)
    assert gate["criterion"] == "f1"
    assert gate["passed"] is passed


def test_rare_positives_switch_to_ranking_gate():
    # 0.2% positives: F1 is zero but the ranking is strong → passes.
    gate = classification_quality_gate(0.0, 0.88, 0.05, 0.002)
    assert gate["criterion"] == "ranking"
    assert gate["passed"] is True
    assert "AP lift=25.0×" in gate["reason"]
    assert gate["positive_rate"] == pytest.approx(0.002)


@pytest.mark.parametrize(
    ("roc_auc", "average_precision"),
    [
        (0.64, 0.05),   # weak ranking, as in the Shell compressor classifier
        (0.90, 0.008),  # AP lift 4× < 5×: strong AUROC but precision near random
    ],
)
def test_ranking_gate_requires_auroc_and_ap_lift(roc_auc, average_precision):
    gate = classification_quality_gate(0.9, roc_auc, average_precision, 0.002)
    assert gate["criterion"] == "ranking"
    assert gate["passed"] is False


def test_ranking_gate_fails_when_ranking_metrics_are_undefined():
    gate = classification_quality_gate(0.0, None, None, 0.002)
    assert gate["passed"] is False
    assert "undefined" in gate["reason"]


@pytest.mark.parametrize(("n_pass", "n_total", "passed"), [(2, 4, True), (1, 3, False), (0, 0, False)])
def test_multilabel_gate_needs_at_least_half_of_labels(n_pass, n_total, passed):
    gates = {f"label_{i}": {"passed": i < n_pass} for i in range(n_total)}
    gate = multilabel_quality_gate(gates)
    assert gate["passed"] is passed
    assert gate["labels_passed"] == n_pass
    assert gate["labels_scored"] == n_total


def test_format_quality_gate():
    assert format_quality_gate({"passed": True, "reason": "x"}) == "Quality gate: ✅ PASSED — x"


class _AllNegativePredictor:
    """Ranks perfectly by ``score`` but predicts all-negative (F1 = 0)."""

    positive_class = 1
    model_best = "FakeModel"
    decision_threshold = 0.5

    def __init__(self, **kwargs):
        pass

    def fit(self, train_data, time_limit=None, presets=None):
        return self

    def predict(self, features):
        return pd.Series([0] * len(features), index=features.index)

    def predict_proba(self, features, as_multiclass=True):
        return pd.Series(features["score"].astype(float).values, index=features.index)

    def evaluate(self, test_df):
        return {"f1": 0.0}

    def leaderboard(self, test_df, silent=True):
        return pd.DataFrame({"model": ["FakeModel"], "score_test": [0.0]})


def test_train_single_gates_rare_failures_on_ranking(monkeypatch, tmp_path):
    """A good ranker at 0.5% positives passes even though default-threshold F1 is 0."""
    monkeypatch.setattr(training, "TabularPredictor", _AllNegativePredictor)
    n = 1000
    train_df = pd.DataFrame({
        "score": [0.1] * (n - 5) + [0.9] * 5,
        "machine_failure": [0] * (n - 5) + [1] * 5,
    })
    test_df = pd.DataFrame({
        "score": [0.1] * 198 + [0.9] * 2,
        "machine_failure": [0] * 198 + [1] * 2,
    })
    args = SimpleNamespace(
        output=tmp_path, time_limit=1, presets="medium_quality", rebalance=False, smote=False,
        seed=42, warm_start=False, skip_importance=True, workers=1, optimization_priority="balanced",
    )

    metrics, _ = training.train_single(train_df, test_df, "classification", args)

    assert metrics["f1"] == 0.0
    assert metrics["positive_rate"] == pytest.approx(0.005)
    assert metrics["quality_gate"]["criterion"] == "ranking"
    assert metrics["quality_gate"]["passed"] is True
