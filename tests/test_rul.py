"""Tests for RUL module — sliding window feature engineering and scoring.

Tests OUR code: sliding_window_features(), nasa_scoring(), formulation detection.
Does NOT test AutoGluon training (that's a benchmark concern).
"""
import numpy as np
import pandas as pd
import pytest

from pdm.rul.model import sliding_window_features, nasa_scoring


class TestSlidingWindowFeatures:
    def test_output_shape(self, rul_df):
        sensor_cols = ["s1", "s2", "s3", "s4"]
        features, indices = sliding_window_features(
            rul_df, unit_col="unit_id", sensor_cols=sensor_cols, window_size=10,
        )
        # 4 sensors × 14 stats = 56 feature columns
        assert features.shape[1] == len(sensor_cols) * 14
        assert len(features) == len(indices)

    def test_correct_row_count(self, rul_df):
        """5 units × (30 - 10 + 1) windows = 105."""
        sensor_cols = ["s1", "s2", "s3", "s4"]
        features, _ = sliding_window_features(
            rul_df, unit_col="unit_id", sensor_cols=sensor_cols, window_size=10,
        )
        assert len(features) == 5 * (30 - 10 + 1)

    def test_deterministic(self, rul_df):
        sensor_cols = ["s1", "s2", "s3", "s4"]
        f1, _ = sliding_window_features(rul_df, "unit_id", sensor_cols, window_size=10)
        f2, _ = sliding_window_features(rul_df, "unit_id", sensor_cols, window_size=10)
        pd.testing.assert_frame_equal(f1, f2)

    def test_no_nans_or_infs(self, rul_df):
        sensor_cols = ["s1", "s2", "s3", "s4"]
        features, _ = sliding_window_features(
            rul_df, unit_col="unit_id", sensor_cols=sensor_cols, window_size=10,
        )
        assert features.isna().sum().sum() == 0
        assert np.isfinite(features.values).all()

    def test_feature_names_pattern(self, rul_df):
        """Names follow sensor_stat convention."""
        sensor_cols = ["s1", "s2"]
        features, _ = sliding_window_features(
            rul_df, unit_col="unit_id", sensor_cols=sensor_cols, window_size=10,
        )
        expected_suffixes = ["_mean", "_std", "_min", "_max", "_last", "_range",
                           "_rms", "_slope", "_kurt", "_p25", "_p75",
                           "_fft_max", "_fft_mean", "_spectral_entropy"]
        for sensor in sensor_cols:
            for suffix in expected_suffixes:
                assert f"{sensor}{suffix}" in features.columns

    def test_stride_reduces_output(self, rul_df):
        sensor_cols = ["s1", "s2", "s3", "s4"]
        f1, _ = sliding_window_features(rul_df, "unit_id", sensor_cols, window_size=10, stride=1)
        f5, _ = sliding_window_features(rul_df, "unit_id", sensor_cols, window_size=10, stride=5)
        assert len(f5) < len(f1)

    def test_larger_window_fewer_rows(self, rul_df):
        sensor_cols = ["s1", "s2"]
        f10, _ = sliding_window_features(rul_df, "unit_id", sensor_cols, window_size=10)
        f20, _ = sliding_window_features(rul_df, "unit_id", sensor_cols, window_size=20)
        assert len(f20) < len(f10)

    def test_single_sensor(self, rul_df):
        features, _ = sliding_window_features(
            rul_df, unit_col="unit_id", sensor_cols=["s1"], window_size=10,
        )
        assert features.shape[1] == 14


class TestNasaScoring:
    def test_perfect_prediction(self):
        y_true = np.array([10, 20, 30])
        y_pred = np.array([10, 20, 30])
        assert nasa_scoring(y_true, y_pred) == 0.0

    def test_late_penalized_more_than_early(self):
        y_true = np.array([50])
        late = nasa_scoring(y_true, np.array([60]))   # d = +10
        early = nasa_scoring(y_true, np.array([40]))  # d = -10
        assert late > early

    def test_asymmetry(self):
        y_true = np.array([50])
        assert nasa_scoring(y_true, np.array([55])) != nasa_scoring(y_true, np.array([45]))

    def test_larger_error_larger_score(self):
        y_true = np.array([50])
        small = nasa_scoring(y_true, np.array([55]))
        large = nasa_scoring(y_true, np.array([65]))
        assert large > small

    def test_multiple_samples_sum(self):
        """Score is sum over samples."""
        y_true = np.array([50, 50])
        y_pred = np.array([55, 55])
        single = nasa_scoring(np.array([50]), np.array([55]))
        assert nasa_scoring(y_true, y_pred) == pytest.approx(2 * single)
