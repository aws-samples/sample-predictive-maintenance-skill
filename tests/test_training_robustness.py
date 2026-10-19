"""Regression tests for class-imbalance training paths.

These tests keep AutoGluon mocked so they exercise the training-data preparation
and predictor wiring without adding model-training time to the unit suite.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pdm.fault_prediction import train as training


class _SpyPredictor:
    """Minimal TabularPredictor stand-in that records constructor and fit inputs."""

    instances = []

    def __init__(self, label, eval_metric, problem_type, path, sample_weight=None):
        self.label = label
        self.eval_metric = eval_metric
        self.problem_type = problem_type
        self.path = path
        self.sample_weight = sample_weight
        self.model_best = "SpyModel"
        self.positive_class = 1
        self.fit_data = None
        self.__class__.instances.append(self)

    def fit(self, train_data, time_limit, presets):
        self.fit_data = train_data.copy()
        return self

    def evaluate(self, test_data):
        return {self.eval_metric: 0.0}

    def leaderboard(self, test_data, silent=True):
        return pd.DataFrame({"model": ["SpyModel"], "score_test": [0.0]})

    def predict(self, features):
        return pd.Series(np.zeros(len(features), dtype=int), index=features.index)

    def predict_proba(self, features, as_multiclass=True):
        positive = pd.Series(np.zeros(len(features), dtype=float), index=features.index)
        if not as_multiclass:
            return positive
        return pd.DataFrame({0: 1.0 - positive, 1: positive}, index=features.index)


@pytest.fixture
def spy_predictor(monkeypatch):
    _SpyPredictor.instances.clear()
    monkeypatch.setattr(training, "TabularPredictor", _SpyPredictor)
    return _SpyPredictor


def _args(tmp_path, **overrides):
    values = {
        "rebalance": False,
        "smote": False,
        "seed": 42,
        "output": tmp_path,
        "warm_start": False,
        "time_limit": 1,
        "presets": "medium_quality",
        "skip_importance": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_rebalance_registers_sample_weight_and_excludes_it_from_features(
    tmp_path, spy_predictor
):
    train_df = pd.DataFrame({
        "sensor_a": np.arange(20, dtype=float),
        "sensor_b": np.linspace(1.0, 2.0, 20),
        "machine_failure": [0] * 16 + [1] * 4,
    })

    _, metadata = training.train_single(
        train_df,
        train_df.copy(),
        "classification",
        _args(tmp_path, rebalance=True),
    )

    predictor = spy_predictor.instances[-1]
    assert predictor.sample_weight == "sample_weight"
    assert "sample_weight" in predictor.fit_data.columns
    assert predictor.fit_data["sample_weight"].mean() == pytest.approx(1.0)
    assert "sample_weight" not in metadata["feature_names"]


def test_smote_imputes_nan_sensor_values_before_resampling(tmp_path, spy_predictor):
    train_df = pd.DataFrame({
        "sensor_a": np.linspace(0.0, 1.0, 18),
        "sensor_b": np.linspace(1.0, 2.0, 18),
        "sensor_all_nan": np.nan,
        "machine_failure": [0] * 12 + [1] * 6,
    })
    train_df.loc[[1, 5, 13], "sensor_a"] = np.nan

    training.train_single(
        train_df,
        train_df.copy(),
        "classification",
        _args(tmp_path, smote=True),
    )

    fitted = spy_predictor.instances[-1].fit_data
    assert not fitted.isna().any().any()
    assert fitted["sensor_all_nan"].eq(0).all()
    counts = fitted["machine_failure"].value_counts()
    assert counts[0] == counts[1]
    assert len(fitted) > len(train_df)
