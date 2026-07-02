"""Tests for AnomalyDetector — Isolation Forest wrapper.

Trains on tiny mock data (100 rows, instant). Tests OUR wrapper logic:
feature selection, threshold, save/load, explain, predict format.
"""
import json
import numpy as np
import pandas as pd
import pytest

from pdm.anomaly_detection.model import AnomalyDetector
from pdm.base import TrainResult, PredictionResult


class TestTrain:
    def test_returns_train_result(self, anomaly_features_df):
        model = AnomalyDetector(contamination=0.1)
        result = model.train(anomaly_features_df, anomaly_features_df)
        assert isinstance(result, TrainResult)

    def test_metrics_contain_threshold(self, anomaly_features_df):
        model = AnomalyDetector(contamination=0.1)
        result = model.train(anomaly_features_df, anomaly_features_df)
        assert "threshold" in result.metrics
        assert result.metrics["threshold"] > 0

    def test_metrics_contain_anomaly_rates(self, anomaly_features_df):
        model = AnomalyDetector(contamination=0.1)
        result = model.train(anomaly_features_df, anomaly_features_df)
        assert 0 <= result.metrics["train_anomaly_rate"] <= 1
        assert 0 <= result.metrics["test_anomaly_rate"] <= 1

    def test_contamination_affects_threshold(self, anomaly_features_df):
        m_low = AnomalyDetector(contamination=0.01)
        m_high = AnomalyDetector(contamination=0.20)
        r_low = m_low.train(anomaly_features_df, anomaly_features_df)
        r_high = m_high.train(anomaly_features_df, anomaly_features_df)
        assert r_low.metrics["threshold"] > r_high.metrics["threshold"]

    def test_feature_selection_excludes_targets(self):
        df = pd.DataFrame({
            "sensor_1": np.random.normal(0, 1, 50),
            "sensor_2": np.random.normal(0, 1, 50),
            "machine_failure": np.zeros(50),
            "RUL": np.arange(50, 0, -1),
            "unit_id": range(50),
        })
        features = AnomalyDetector._select_features(df)
        assert "machine_failure" not in features
        assert "RUL" not in features
        assert "unit_id" not in features
        assert "sensor_1" in features


class TestPredict:
    def test_returns_prediction_result(self, anomaly_features_df):
        model = AnomalyDetector(contamination=0.1)
        model.train(anomaly_features_df, anomaly_features_df)
        result = model.predict(anomaly_features_df)
        assert isinstance(result, PredictionResult)

    def test_output_columns(self, anomaly_features_df):
        model = AnomalyDetector(contamination=0.1)
        model.train(anomaly_features_df, anomaly_features_df)
        result = model.predict(anomaly_features_df)
        assert "anomaly_score" in result.predictions.columns
        assert "is_anomaly" in result.predictions.columns

    def test_shape_matches_input(self, anomaly_features_df):
        model = AnomalyDetector(contamination=0.1)
        model.train(anomaly_features_df, anomaly_features_df)
        result = model.predict(anomaly_features_df)
        assert len(result.predictions) == len(anomaly_features_df)

    def test_is_anomaly_binary(self, anomaly_features_df):
        model = AnomalyDetector(contamination=0.1)
        model.train(anomaly_features_df, anomaly_features_df)
        result = model.predict(anomaly_features_df)
        assert set(result.predictions["is_anomaly"].unique()).issubset({0, 1})

    def test_detects_injected_spikes(self, anomaly_df, anomaly_features_df):
        """Spikes injected in conftest should be partially detected."""
        model = AnomalyDetector(contamination=0.1)
        model.train(anomaly_features_df, anomaly_features_df)
        result = model.predict(anomaly_features_df)
        assert result.predictions["is_anomaly"].sum() > 0

    def test_does_not_mutate_input(self, anomaly_features_df):
        model = AnomalyDetector(contamination=0.1)
        model.train(anomaly_features_df, anomaly_features_df)
        original = anomaly_features_df.copy()
        model.predict(anomaly_features_df)
        pd.testing.assert_frame_equal(anomaly_features_df, original)


class TestExplain:
    def test_returns_list_of_lists(self, anomaly_features_df):
        model = AnomalyDetector(contamination=0.1)
        model.train(anomaly_features_df, anomaly_features_df)
        explanations = model.explain(anomaly_features_df.head(3), top_k=3)
        assert isinstance(explanations, list)
        assert len(explanations) == 3
        for exp in explanations:
            assert isinstance(exp, list)
            for item in exp:
                assert "feature" in item
                assert "contribution" in item


class TestSaveLoad:
    def test_save_creates_files(self, anomaly_features_df, tmp_model_dir):
        model = AnomalyDetector(contamination=0.1)
        model.train(anomaly_features_df, anomaly_features_df)
        model.save(tmp_model_dir)
        assert (tmp_model_dir / "isolation_forest.joblib").exists()
        assert (tmp_model_dir / "scaler.joblib").exists()
        assert (tmp_model_dir / "metadata.json").exists()

    def test_roundtrip_scores_match(self, anomaly_features_df, tmp_model_dir):
        model = AnomalyDetector(contamination=0.1)
        model.train(anomaly_features_df, anomaly_features_df)
        preds_before = model.predict(anomaly_features_df)
        model.save(tmp_model_dir)

        loaded = AnomalyDetector.load(tmp_model_dir)
        preds_after = loaded.predict(anomaly_features_df)
        np.testing.assert_array_almost_equal(
            preds_before.predictions["anomaly_score"].values,
            preds_after.predictions["anomaly_score"].values,
        )

    def test_loaded_attributes(self, anomaly_features_df, tmp_model_dir):
        model = AnomalyDetector(contamination=0.1)
        model.train(anomaly_features_df, anomaly_features_df)
        model.save(tmp_model_dir)

        loaded = AnomalyDetector.load(tmp_model_dir)
        assert loaded.feature_names == model.feature_names
        assert loaded.threshold == model.threshold

    def test_metadata_json_valid(self, anomaly_features_df, tmp_model_dir):
        model = AnomalyDetector(contamination=0.1)
        model.train(anomaly_features_df, anomaly_features_df)
        model.save(tmp_model_dir)

        meta = json.loads((tmp_model_dir / "metadata.json").read_text())
        assert meta["formulation"] == "anomaly_detection"
        assert "feature_names" in meta
        assert "threshold" in meta
