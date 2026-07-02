"""Tests for PDMModel ABC — data structures, formulation detection, model registry."""
import pandas as pd
import pytest

from pdm.base import PDMModel, TrainResult, PredictionResult


class TestTrainResult:
    def test_required_fields(self):
        r = TrainResult(model="dummy", metrics={"rmse": 1.0})
        assert r.model == "dummy"
        assert r.metrics["rmse"] == 1.0
        assert r.feature_importance is None
        assert r.metadata == {}

    def test_optional_fields(self):
        importance = pd.Series({"a": 0.5, "b": 0.3})
        r = TrainResult(model="x", metrics={}, feature_importance=importance,
                       metadata={"formulation": "rul"})
        assert r.feature_importance is not None
        assert r.metadata["formulation"] == "rul"


class TestPredictionResult:
    def test_minimal(self):
        r = PredictionResult(predictions=pd.DataFrame({"y": [1, 2, 3]}))
        assert len(r.predictions) == 3
        assert r.explanations is None

    def test_backwards_compatible_no_intervals(self):
        """Old PredictionResult (before P1 uncertainty) still works."""
        r = PredictionResult(predictions=pd.DataFrame({"y": [1]}))
        assert getattr(r, "intervals", None) is None

    def test_with_explanations(self):
        r = PredictionResult(
            predictions=pd.DataFrame({"y": [1]}),
            explanations=[{"sensor_1": 0.5}],
        )
        assert len(r.explanations) == 1


class TestFormulationDetection:
    def test_rul(self, rul_df):
        assert PDMModel.detect_formulation(rul_df) == "rul"

    def test_classification(self, classification_df):
        assert PDMModel.detect_formulation(classification_df) == "classification"

    def test_survival(self, survival_df):
        assert PDMModel.detect_formulation(survival_df) == "survival"

    def test_multilabel(self):
        df = pd.DataFrame({"label_a": [0, 1], "label_b": [1, 0], "f": [1, 2]})
        assert PDMModel.detect_formulation(df) == "multilabel"

    def test_multilabel_priority_over_rul(self):
        df = pd.DataFrame({"label_a": [0], "RUL": [10], "f": [1]})
        assert PDMModel.detect_formulation(df) == "multilabel"

    def test_rul_priority_over_classification(self):
        df = pd.DataFrame({"RUL": [10], "machine_failure": [0], "f": [1]})
        assert PDMModel.detect_formulation(df) == "rul"

    def test_survival_needs_both_columns(self):
        df = pd.DataFrame({"duration": [100], "f": [1]})
        with pytest.raises(ValueError):
            PDMModel.detect_formulation(df)

    def test_unknown_raises(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        with pytest.raises(ValueError, match="Cannot detect"):
            PDMModel.detect_formulation(df)


class TestModelRegistry:
    def test_rul(self):
        assert PDMModel.get_model_class("rul").__name__ == "RULPredictor"

    def test_classification(self):
        assert PDMModel.get_model_class("classification").__name__ == "FailureClassifier"

    def test_multilabel(self):
        assert PDMModel.get_model_class("multilabel").__name__ == "FailureClassifier"

    def test_survival(self):
        assert PDMModel.get_model_class("survival").__name__ == "SurvivalPredictor"

    def test_anomaly(self):
        assert PDMModel.get_model_class("anomaly_detection").__name__ == "AnomalyDetector"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            PDMModel.get_model_class("bogus")
