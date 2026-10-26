"""Tests for SpectralResidualDetector — frequency-domain saliency anomaly detection.

Uses mock time-series data (200 timesteps, 5 features) with injected anomaly segments.
"""
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from pdm.anomaly_detection.spectral_residual import SpectralResidualDetector
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
        model = SpectralResidualDetector(sr_window=5, aggregation_percentile=95)
        result = model.train(temporal_train_df, temporal_test_df)
        assert isinstance(result, TrainResult)

    def test_metrics_contain_expected_keys(self, temporal_train_df, temporal_test_df):
        model = SpectralResidualDetector(sr_window=5, aggregation_percentile=95)
        result = model.train(temporal_train_df, temporal_test_df)
        assert "threshold" in result.metrics
        assert "sr_window" in result.metrics
        assert "aggregation_percentile" in result.metrics

    def test_feature_selection_excludes_label(self, temporal_train_df, temporal_test_df):
        model = SpectralResidualDetector(sr_window=5, aggregation_percentile=95)
        model.train(temporal_train_df, temporal_test_df)
        assert "label" not in model.feature_names
        assert all(f.startswith("sensor_") for f in model.feature_names)

    def test_train_sr_stats_computed(self, temporal_train_df, temporal_test_df):
        model = SpectralResidualDetector(sr_window=5)
        model.train(temporal_train_df, temporal_test_df)
        assert model._train_sr_mean is not None
        assert model._train_sr_std is not None
        assert len(model._train_sr_mean) == 5  # 5 features
        assert len(model._train_sr_std) == 5


class TestPredict:
    def test_returns_prediction_result(self, temporal_train_df, temporal_test_df):
        model = SpectralResidualDetector(sr_window=5)
        model.train(temporal_train_df, temporal_test_df)
        feature_cols = model.feature_names
        result = model.predict(temporal_test_df[feature_cols])
        assert isinstance(result, PredictionResult)

    def test_output_columns(self, temporal_train_df, temporal_test_df):
        model = SpectralResidualDetector(sr_window=5)
        model.train(temporal_train_df, temporal_test_df)
        result = model.predict(temporal_test_df[model.feature_names])
        assert "anomaly_score" in result.predictions.columns
        assert "is_anomaly" in result.predictions.columns

    def test_shape_matches_input(self, temporal_train_df, temporal_test_df):
        model = SpectralResidualDetector(sr_window=5)
        model.train(temporal_train_df, temporal_test_df)
        result = model.predict(temporal_test_df[model.feature_names])
        assert len(result.predictions) == len(temporal_test_df)

    def test_is_anomaly_binary(self, temporal_train_df, temporal_test_df):
        model = SpectralResidualDetector(sr_window=5)
        model.train(temporal_train_df, temporal_test_df)
        result = model.predict(temporal_test_df[model.feature_names])
        assert set(result.predictions["is_anomaly"].unique()).issubset({0, 1})

    def test_anomaly_segment_scores_rank_above_normal(self, temporal_train_df, temporal_test_df):
        """Injected spikes should rank above normal points without test-label calibration."""
        model = SpectralResidualDetector(sr_window=5, contamination=0.2)
        model.train(temporal_train_df, temporal_test_df)
        result = model.predict(temporal_test_df[model.feature_names])
        auc = roc_auc_score(
            temporal_test_df["label"].values,
            result.predictions["anomaly_score"].values,
        )
        assert auc > 0.9


class TestExplain:
    def test_returns_list(self, temporal_train_df, temporal_test_df):
        model = SpectralResidualDetector(sr_window=5)
        model.train(temporal_train_df, temporal_test_df)
        explanations = model.explain(temporal_test_df[model.feature_names].head(3), top_k=3)
        assert isinstance(explanations, list)
        assert len(explanations) == 3

    def test_explanation_format(self, temporal_train_df, temporal_test_df):
        model = SpectralResidualDetector(sr_window=5)
        model.train(temporal_train_df, temporal_test_df)
        explanations = model.explain(temporal_test_df[model.feature_names].head(1), top_k=3)
        for item in explanations[0]:
            assert "feature" in item
            assert "contribution" in item


class TestSaveLoad:
    def test_save_creates_files(self, temporal_train_df, temporal_test_df, tmp_path):
        model_dir = tmp_path / "model"
        model = SpectralResidualDetector(sr_window=5)
        model.train(temporal_train_df, temporal_test_df)
        model.save(model_dir)
        assert (model_dir / "scaler.joblib").exists()
        assert (model_dir / "metadata.json").exists()
        assert (model_dir / "train_sr_mean.npy").exists()
        assert (model_dir / "train_sr_std.npy").exists()

    def test_roundtrip_scores_match(self, temporal_train_df, temporal_test_df, tmp_path):
        model_dir = tmp_path / "model"
        model = SpectralResidualDetector(sr_window=5)
        model.train(temporal_train_df, temporal_test_df)
        preds_before = model.predict(temporal_test_df[model.feature_names])
        model.save(model_dir)

        loaded = SpectralResidualDetector.load(model_dir)
        preds_after = loaded.predict(temporal_test_df[loaded.feature_names])
        np.testing.assert_array_almost_equal(
            preds_before.predictions["anomaly_score"].values,
            preds_after.predictions["anomaly_score"].values,
        )

    def test_metadata_json_method(self, temporal_train_df, temporal_test_df, tmp_path):
        model_dir = tmp_path / "model"
        model = SpectralResidualDetector(sr_window=5, aggregation_percentile=95)
        model.train(temporal_train_df, temporal_test_df)
        model.save(model_dir)

        meta = json.loads((model_dir / "metadata.json").read_text())
        assert meta["method"] == "spectral_residual"
        assert meta["sr_window"] == 5
        assert meta["aggregation_percentile"] == 95

    def test_metadata_records_block_size_and_score_version(
        self, temporal_train_df, temporal_test_df, tmp_path
    ):
        from pdm.anomaly_detection.spectral_residual import SCORE_VERSION

        model_dir = tmp_path / "model"
        model = SpectralResidualDetector(sr_window=5, sr_block_size=64)
        model.train(temporal_train_df, temporal_test_df)
        model.save(model_dir)

        meta = json.loads((model_dir / "metadata.json").read_text())
        assert meta["sr_block_size"] == 64
        assert meta["score_version"] == SCORE_VERSION
        assert SpectralResidualDetector.load(model_dir).sr_block_size == 64

    def test_load_rejects_legacy_score_scale(self, temporal_train_df, temporal_test_df, tmp_path):
        """Artifacts from the old global-FFT scoring must be retrained, not reused."""
        model_dir = tmp_path / "model"
        model = SpectralResidualDetector(sr_window=5)
        model.train(temporal_train_df, temporal_test_df)
        model.save(model_dir)

        meta_path = model_dir / "metadata.json"
        meta = json.loads(meta_path.read_text())
        del meta["score_version"]
        del meta["sr_block_size"]
        meta_path.write_text(json.dumps(meta))

        with pytest.raises(ValueError, match="score_version"):
            SpectralResidualDetector.load(model_dir)


def _normal_process(n: int, seed: int) -> pd.DataFrame:
    """Stationary normal-only process with a noisy sinusoid and an exactly periodic
    binary channel (the latter has exactly-zero FFT bins, like many SMAP channels)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return pd.DataFrame({
        "sensor_sine": np.sin(2 * np.pi * t / 37) + rng.normal(0, 0.3, n),
        "sensor_noise": rng.normal(0, 1, n),
        "sensor_binary": ((t // 8) % 2).astype(float),
    })


class TestScoreScale:
    """Scores must be on the same scale regardless of input length (regression for
    the global-FFT implementation whose saliency scaled with 1/N and whose absolute
    log floor exploded next to exactly-zero spectral bins)."""

    CONTAMINATION = 0.10

    @pytest.fixture
    def trained(self):
        train_df = _normal_process(2048, seed=0)
        model = SpectralResidualDetector(
            sr_window=5, contamination=self.CONTAMINATION, sr_block_size=256
        )
        model.train(train_df, _normal_process(512, seed=1))
        return model

    def test_longer_normal_series_alert_rate_matches_contamination(self, trained):
        """A 4x longer normal series must alert at roughly the configured rate."""
        test_df = _normal_process(8192, seed=2)
        alert_rate = trained.predict(test_df[trained.feature_names]).predictions["is_anomaly"].mean()
        assert self.CONTAMINATION / 2 < alert_rate < self.CONTAMINATION * 2

    def test_block_aligned_batches_match_full_series_scores(self, trained):
        """Scoring in block-sized batches (streaming) equals scoring the whole series."""
        test_df = _normal_process(2048, seed=3)[trained.feature_names]
        full = trained.predict(test_df).predictions["anomaly_score"].values
        batched = np.concatenate([
            trained.predict(test_df.iloc[i:i + 256]).predictions["anomaly_score"].values
            for i in range(0, len(test_df), 256)
        ])
        np.testing.assert_allclose(batched, full)

    def test_short_batches_do_not_explode(self, trained):
        """Batches shorter than the block stay on the calibrated scale."""
        test_df = _normal_process(2048, seed=4)[trained.feature_names]
        batched = np.concatenate([
            trained.predict(test_df.iloc[i:i + 64]).predictions["is_anomaly"].values
            for i in range(0, len(test_df), 64)
        ])
        assert batched.mean() < self.CONTAMINATION * 3

    def test_short_trailing_remainder_merged_into_previous_block(self):
        data = np.random.default_rng(0).normal(size=(256 + 10, 2))
        merged = SpectralResidualDetector._spectral_residual_per_feature(data, 5, 256)
        single = SpectralResidualDetector._spectral_residual_per_feature(data, 5, 10_000)
        np.testing.assert_allclose(merged, single)

    def test_invalid_block_size_rejected(self):
        with pytest.raises(ValueError, match="sr_block_size"):
            SpectralResidualDetector(sr_block_size=1)
