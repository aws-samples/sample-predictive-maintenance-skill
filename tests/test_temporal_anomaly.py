"""Tests for TemporalAnomalyDetector — sliding-window PCA reconstruction.

Uses mock time-series data (200 timesteps, 5 features) with injected anomaly segments.
"""
import json
import numpy as np
import pandas as pd
import pytest

from pdm.anomaly_detection.temporal import TemporalAnomalyDetector
from pdm.base import TrainResult, PredictionResult


@pytest.fixture
def temporal_data():
    """200 timesteps, 5 features. Normal = sine waves. Anomaly segment at t=150-170."""
    np.random.seed(42)
    n = 200
    t = np.arange(n)
    features = np.column_stack([
        np.sin(t * 0.1) + np.random.normal(0, 0.05, n),
        np.cos(t * 0.15) + np.random.normal(0, 0.05, n),
        np.sin(t * 0.2 + 1) + np.random.normal(0, 0.05, n),
        np.cos(t * 0.05) + np.random.normal(0, 0.05, n),
        np.sin(t * 0.3) + np.random.normal(0, 0.05, n),
    ])
    cols = [f"sensor_{i}" for i in range(5)]
    df = pd.DataFrame(features, columns=cols)
    return df


@pytest.fixture
def temporal_train_df(temporal_data):
    """First 100 timesteps (all normal)."""
    return temporal_data.iloc[:100].reset_index(drop=True)


@pytest.fixture
def temporal_test_df(temporal_data):
    """Last 100 timesteps with anomaly segment (spikes at t=50-70)."""
    test = temporal_data.iloc[100:].reset_index(drop=True).copy()
    # Inject anomaly: large spikes
    test.iloc[50:70, :] += np.random.normal(5, 1, (20, 5))
    # Add label
    test["label"] = 0
    test.loc[50:69, "label"] = 1
    return test


class TestTrain:
    def test_returns_train_result(self, temporal_train_df, temporal_test_df):
        model = TemporalAnomalyDetector(window_size=3, smooth_window=5)
        result = model.train(temporal_train_df, temporal_test_df)
        assert isinstance(result, TrainResult)

    def test_metrics_contain_expected_keys(self, temporal_train_df, temporal_test_df):
        model = TemporalAnomalyDetector(window_size=3, smooth_window=5)
        result = model.train(temporal_train_df, temporal_test_df)
        assert "threshold" in result.metrics
        assert "pca_components" in result.metrics
        assert "window_size" in result.metrics
        assert "smooth_window" in result.metrics

    def test_feature_selection_excludes_label(self, temporal_train_df, temporal_test_df):
        model = TemporalAnomalyDetector(window_size=3, smooth_window=5)
        model.train(temporal_train_df, temporal_test_df)
        assert "label" not in model.feature_names
        assert all(f.startswith("sensor_") for f in model.feature_names)

    def test_threshold_positive(self, temporal_train_df, temporal_test_df):
        model = TemporalAnomalyDetector(window_size=3, smooth_window=5)
        result = model.train(temporal_train_df, temporal_test_df)
        assert result.metrics["threshold"] > 0


class TestPredict:
    def test_returns_prediction_result(self, temporal_train_df, temporal_test_df):
        model = TemporalAnomalyDetector(window_size=3, smooth_window=5)
        model.train(temporal_train_df, temporal_test_df)
        feature_cols = model.feature_names
        result = model.predict(temporal_test_df[feature_cols])
        assert isinstance(result, PredictionResult)

    def test_output_columns(self, temporal_train_df, temporal_test_df):
        model = TemporalAnomalyDetector(window_size=3, smooth_window=5)
        model.train(temporal_train_df, temporal_test_df)
        result = model.predict(temporal_test_df[model.feature_names])
        assert "anomaly_score" in result.predictions.columns
        assert "is_anomaly" in result.predictions.columns

    def test_shape_matches_input(self, temporal_train_df, temporal_test_df):
        model = TemporalAnomalyDetector(window_size=3, smooth_window=5)
        model.train(temporal_train_df, temporal_test_df)
        result = model.predict(temporal_test_df[model.feature_names])
        assert len(result.predictions) == len(temporal_test_df)

    def test_is_anomaly_binary(self, temporal_train_df, temporal_test_df):
        model = TemporalAnomalyDetector(window_size=3, smooth_window=5)
        model.train(temporal_train_df, temporal_test_df)
        result = model.predict(temporal_test_df[model.feature_names])
        assert set(result.predictions["is_anomaly"].unique()).issubset({0, 1})

    def test_detects_anomaly_segment(self, temporal_train_df, temporal_test_df):
        """The injected spike segment should be at least partially detected."""
        model = TemporalAnomalyDetector(window_size=3, smooth_window=5, contamination=0.2)
        model.train(temporal_train_df, temporal_test_df)
        result = model.predict(temporal_test_df[model.feature_names])
        # At least some anomalies should be flagged in the spike region (50-70)
        anomaly_in_segment = result.predictions["is_anomaly"].iloc[50:70].sum()
        assert anomaly_in_segment > 0


class TestExplain:
    def test_returns_list(self, temporal_train_df, temporal_test_df):
        model = TemporalAnomalyDetector(window_size=3, smooth_window=5)
        model.train(temporal_train_df, temporal_test_df)
        explanations = model.explain(temporal_test_df[model.feature_names].head(3), top_k=3)
        assert isinstance(explanations, list)
        assert len(explanations) == 3

    def test_explanation_format(self, temporal_train_df, temporal_test_df):
        model = TemporalAnomalyDetector(window_size=3, smooth_window=5)
        model.train(temporal_train_df, temporal_test_df)
        explanations = model.explain(temporal_test_df[model.feature_names].head(1), top_k=3)
        for item in explanations[0]:
            assert "feature" in item
            assert "contribution" in item


class TestSaveLoad:
    def test_save_creates_files(self, temporal_train_df, temporal_test_df, tmp_path):
        model_dir = tmp_path / "model"
        model = TemporalAnomalyDetector(window_size=3, smooth_window=5)
        model.train(temporal_train_df, temporal_test_df)
        model.save(model_dir)
        assert (model_dir / "pca_model.joblib").exists()
        assert (model_dir / "scaler.joblib").exists()
        assert (model_dir / "metadata.json").exists()
        assert (model_dir / "train_error_mean.npy").exists()
        assert (model_dir / "train_error_std.npy").exists()

    def test_roundtrip_scores_match(self, temporal_train_df, temporal_test_df, tmp_path):
        model_dir = tmp_path / "model"
        model = TemporalAnomalyDetector(window_size=3, smooth_window=5)
        model.train(temporal_train_df, temporal_test_df)
        preds_before = model.predict(temporal_test_df[model.feature_names])
        model.save(model_dir)

        loaded = TemporalAnomalyDetector.load(model_dir)
        preds_after = loaded.predict(temporal_test_df[loaded.feature_names])
        np.testing.assert_array_almost_equal(
            preds_before.predictions["anomaly_score"].values,
            preds_after.predictions["anomaly_score"].values,
        )

    def test_metadata_json_method(self, temporal_train_df, temporal_test_df, tmp_path):
        model_dir = tmp_path / "model"
        model = TemporalAnomalyDetector(window_size=3, smooth_window=5)
        model.train(temporal_train_df, temporal_test_df)
        model.save(model_dir)

        meta = json.loads((model_dir / "metadata.json").read_text())
        assert meta["method"] == "temporal_pca_reconstruction"
        assert meta["window_size"] == 3
        assert meta["smooth_window"] == 5
