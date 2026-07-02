"""Tests for deployment contracts — predict format, JSON serialization."""
import json
import numpy as np
import pandas as pd
import pytest

from pdm.anomaly_detection.model import AnomalyDetector
from pdm.base import PredictionResult


class TestPredictContract:
    """Guards against P0 (drift check) and P1 (intervals) breaking output format."""

    def test_predict_returns_dataframe(self, anomaly_features_df):
        model = AnomalyDetector(contamination=0.1)
        model.train(anomaly_features_df, anomaly_features_df)
        result = model.predict(anomaly_features_df)
        assert isinstance(result.predictions, pd.DataFrame)

    def test_predictions_json_serializable(self, anomaly_features_df):
        model = AnomalyDetector(contamination=0.1)
        model.train(anomaly_features_df, anomaly_features_df)
        result = model.predict(anomaly_features_df.head(5))

        response = {
            "prediction": result.predictions["is_anomaly"].tolist(),
            "scores": result.predictions["anomaly_score"].tolist(),
        }
        serialized = json.dumps(response)
        parsed = json.loads(serialized)
        assert len(parsed["prediction"]) == 5
        assert all(isinstance(v, (int, float)) for v in parsed["scores"])

    def test_backwards_compatible_no_intervals(self, anomaly_features_df):
        """Old API response (no intervals field) still works."""
        model = AnomalyDetector(contamination=0.1)
        model.train(anomaly_features_df, anomaly_features_df)
        result = model.predict(anomaly_features_df.head(1))
        assert getattr(result, "intervals", None) is None
