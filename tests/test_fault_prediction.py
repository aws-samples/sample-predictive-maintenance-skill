"""Tests for FailureClassifier — binary/multilabel detection and interface.

Uses very short time_limit (10s) on tiny data (100 rows) for fast training.
"""
import numpy as np
import pandas as pd
import pytest

from pdm.fault_prediction.model import FailureClassifier
from pdm.base import TrainResult, PredictionResult


class TestFormulationDetection:
    def test_binary_detected(self, classification_df):
        model = FailureClassifier()
        model.train(classification_df, classification_df, time_limit=10)
        assert model._is_multilabel is False
        assert model.label_names == ["machine_failure"]

    def test_multilabel_detected(self):
        df = pd.DataFrame({
            "f1": np.random.normal(0, 1, 50),
            "f2": np.random.normal(0, 1, 50),
            "label_a": (np.random.random(50) < 0.2).astype(int),
            "label_b": (np.random.random(50) < 0.15).astype(int),
        })
        model = FailureClassifier()
        model.train(df, df, time_limit=10)
        assert model._is_multilabel is True
        assert "label_a" in model.label_names
        assert "label_b" in model.label_names

    def test_no_target_raises(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        model = FailureClassifier()
        with pytest.raises(ValueError, match="No target found"):
            model.train(df, df, time_limit=10)


class TestFeatureExclusion:
    def test_target_excluded_from_features(self, classification_df):
        model = FailureClassifier()
        model.train(classification_df, classification_df, time_limit=10)
        assert "machine_failure" not in model.feature_names
        assert "air_temp" in model.feature_names
        assert "torque" in model.feature_names


class TestTrainAndPredict:
    def test_train_returns_result(self, classification_df):
        model = FailureClassifier()
        result = model.train(classification_df, classification_df, time_limit=10)
        assert isinstance(result, TrainResult)
        assert len(result.metrics) > 0

    def test_predict_returns_result(self, classification_df):
        model = FailureClassifier()
        model.train(classification_df, classification_df, time_limit=10)
        preds = model.predict(classification_df)
        assert isinstance(preds, PredictionResult)
        assert len(preds.predictions) > 0
