"""Regression coverage for ranking metrics emitted by the training CLI."""

from types import SimpleNamespace

import pandas as pd
import pytest

from pdm.fault_prediction import train as training


class _FakePredictor:
    positive_class = 1
    model_best = "FakeModel"
    decision_threshold = 0.37

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, train_data, time_limit=None, presets=None):
        return self

    def predict(self, features):
        return pd.Series([0] * len(features), index=features.index)

    def predict_proba(self, features, as_multiclass=True):
        assert as_multiclass is False
        return pd.Series(features["score"].astype(float).values, index=features.index)

    def evaluate(self, test_df):
        return {"f1": 0.0}

    def leaderboard(self, test_df, silent=True):
        return pd.DataFrame({"model": ["FakeModel"], "score_test": [0.0]})


def _args(tmp_path):
    return SimpleNamespace(
        output=tmp_path,
        time_limit=1,
        presets="medium_quality",
        rebalance=False,
        smote=False,
        seed=42,
        warm_start=False,
        skip_importance=True,
        workers=1,
        optimization_priority="balanced",
    )


def test_binary_training_reports_ranking_metrics(monkeypatch, tmp_path):
    monkeypatch.setattr(training, "TabularPredictor", _FakePredictor)
    train_df = pd.DataFrame({
        "score": [0.05, 0.10, 0.15, 0.20, 0.80, 0.85, 0.90, 0.95] * 2,
        "machine_failure": [0, 0, 0, 0, 1, 1, 1, 1] * 2,
    })
    test_df = pd.DataFrame({
        "score": [0.10, 0.20, 0.80, 0.90],
        "machine_failure": [0, 0, 1, 1],
    })

    metrics, _ = training.train_single(
        train_df, test_df, "classification", _args(tmp_path)
    )

    assert metrics["test_score"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["optimization_priority"] == "balanced"
    assert metrics["decision_threshold"] == pytest.approx(0.37)
    # 50% positives → F1 gate; the fake predictor predicts all-negative, so it fails.
    assert metrics["positive_rate"] == pytest.approx(0.5)
    assert metrics["quality_gate"]["criterion"] == "f1"
    assert metrics["quality_gate"]["passed"] is False


def test_multilabel_training_reports_per_label_ranking_metrics(monkeypatch, tmp_path):
    monkeypatch.setattr(training, "TabularPredictor", _FakePredictor)
    train_df = pd.DataFrame({
        "score": [0.10] * 10 + [0.90] * 10,
        "label_failure": [0] * 10 + [1] * 10,
    })
    test_df = pd.DataFrame({
        "score": [0.10, 0.20, 0.80, 0.90],
        "label_failure": [0, 0, 1, 1],
    })

    _, result, _ = training._train_single_label(
        "label_failure",
        ["score"],
        train_df,
        test_df,
        tmp_path,
        1,
        "medium_quality",
        skip_importance=True,
    )

    assert result["test_f1"] == 0.0
    assert result["test_roc_auc"] == pytest.approx(1.0)
    assert result["test_average_precision"] == pytest.approx(1.0)
    assert result["optimization_priority"] == "balanced"
    assert result["decision_threshold"] == pytest.approx(0.37)
