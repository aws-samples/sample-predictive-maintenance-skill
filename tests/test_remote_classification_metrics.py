"""Tests that SageMaker training entry points emit ranking metrics."""

import json

import pandas as pd
import pytest

from pdm.remote import train_multilabel, train_remote


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


def _patch_autogluon(monkeypatch):
    import autogluon.tabular as ag_tabular

    monkeypatch.setattr(ag_tabular, "TabularPredictor", _FakePredictor)


def test_remote_binary_training_emits_ranking_metrics(monkeypatch, tmp_path):
    _patch_autogluon(monkeypatch)
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "output"
    data_dir.mkdir()

    train_df = pd.DataFrame({
        "score": [0.10] * 10 + [0.90] * 10,
        "machine_failure": [0] * 10 + [1] * 10,
    })
    test_df = pd.DataFrame({
        "score": [0.10, 0.20, 0.80, 0.90],
        "machine_failure": [0, 0, 1, 1],
    })
    train_df.to_csv(data_dir / "raw_train.csv", index=False)
    test_df.to_csv(data_dir / "raw_test.csv", index=False)

    monkeypatch.setenv(
        "SM_HPS",
        json.dumps({
            "formulation": "classification",
            "time-limit": 1,
            "optimization-priority": "recall",
        }),
    )
    monkeypatch.setenv("SM_CHANNEL_TRAIN", str(data_dir))
    monkeypatch.setenv("SM_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("SM_OUTPUT_DATA_DIR", str(output_dir))

    train_remote.main()

    metrics = json.loads((model_dir / "metrics.json").read_text())
    assert metrics["f1"] == 0.0
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["optimization_priority"] == "recall"
    assert metrics["eval_metric"] == "f2"
    assert metrics["decision_threshold"] == pytest.approx(0.37)


def test_remote_multilabel_training_emits_ranking_metrics(monkeypatch, tmp_path, capsys):
    _patch_autogluon(monkeypatch)
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "model"
    data_dir.mkdir()

    train_df = pd.DataFrame({
        "score": [0.10] * 10 + [0.90] * 10,
        "label_failure": [0] * 10 + [1] * 10,
    })
    test_df = pd.DataFrame({
        "score": [0.10, 0.20, 0.80, 0.90],
        "label_failure": [0, 0, 1, 1],
    })
    train_df.to_csv(data_dir / "train.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)

    monkeypatch.setenv(
        "SM_HPS",
        json.dumps({
            "time-limit": 1,
            "presets": "medium_quality",
            "workers": 1,
            "optimization-priority": "precision",
        }),
    )
    monkeypatch.setenv("SM_CHANNEL_TRAIN", str(data_dir))
    monkeypatch.setenv("SM_MODEL_DIR", str(model_dir))

    train_multilabel.main()

    metrics = json.loads((model_dir / "metrics.json").read_text())
    label_metrics = metrics["labels"]["label_failure"]
    assert label_metrics["test_f1"] == 0.0
    assert label_metrics["test_roc_auc"] == pytest.approx(1.0)
    assert label_metrics["test_average_precision"] == pytest.approx(1.0)
    assert metrics["mean_test_roc_auc"] == pytest.approx(1.0)
    assert metrics["mean_test_average_precision"] == pytest.approx(1.0)
    assert metrics["optimization_priority"] == "precision"
    assert label_metrics["eval_metric"] == "f0.5"
    assert label_metrics["decision_threshold"] == pytest.approx(0.37)

    output = capsys.readouterr().out
    assert "✅ Metrics: mean_f1=0.0, median_f1=0.0" in output
    assert "Ranking metrics: mean_roc_auc=1.0, mean_average_precision=1.0" in output
