"""Tests for pdm/monitoring — baseline capture and drift detection."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from pdm.monitoring.baseline import capture_baseline, save_baseline, load_baseline, FeatureBaseline
from pdm.monitoring.drift import detect_drift, DriftSeverity, DriftReport


class TestBaseline:
    def test_capture_returns_list(self, anomaly_features_df):
        baselines = capture_baseline(anomaly_features_df, ["sensor_1", "sensor_2"])
        assert len(baselines) == 2
        assert all(isinstance(b, FeatureBaseline) for b in baselines)

    def test_capture_stats_correct(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        baselines = capture_baseline(df, ["x"])
        b = baselines[0]
        assert b.name == "x"
        assert b.mean == 3.0
        assert b.min == 1.0
        assert b.max == 5.0
        assert b.n_samples == 5

    def test_save_load_roundtrip(self, tmp_path):
        baselines = [FeatureBaseline(
            name="f1", mean=0.0, std=1.0, min=-3.0, max=3.0,
            p5=-1.6, p25=-0.7, p50=0.0, p75=0.7, p95=1.6, n_samples=100,
        )]
        path = tmp_path / "baseline.json"
        save_baseline(baselines, path)
        loaded = load_baseline(path)
        assert loaded[0].name == "f1"
        assert loaded[0].mean == 0.0
        assert loaded[0].n_samples == 100


class TestDriftDetection:
    def test_no_drift_on_same_distribution(self):
        """Same distribution should not trigger drift."""
        rng = np.random.default_rng(42)
        df_ref = pd.DataFrame({"x": rng.normal(0, 1, 200)})
        baselines = capture_baseline(df_ref, ["x"])

        df_new = pd.DataFrame({"x": rng.normal(0, 1, 100)})
        report = detect_drift(df_new, baselines)
        assert report.is_healthy
        assert report.overall_severity == DriftSeverity.NONE

    def test_drift_detected_on_shifted_distribution(self):
        """Mean shift of 3 std should trigger critical drift."""
        rng = np.random.default_rng(42)
        df_ref = pd.DataFrame({"x": rng.normal(0, 1, 200)})
        baselines = capture_baseline(df_ref, ["x"])

        df_shifted = pd.DataFrame({"x": rng.normal(5, 1, 100)})  # shifted by 5 std
        report = detect_drift(df_shifted, baselines)
        assert not report.is_healthy
        assert report.n_drifted_critical > 0

    def test_mean_shift_computed(self):
        """mean_shift should reflect (new_mean - ref_mean) / ref_std."""
        baselines = [FeatureBaseline(
            name="x", mean=10.0, std=2.0, min=5.0, max=15.0,
            p5=6.7, p25=8.6, p50=10.0, p75=11.3, p95=13.3, n_samples=200,
        )]
        df = pd.DataFrame({"x": np.full(50, 14.0)})  # mean=14, shift=(14-10)/2=2.0
        report = detect_drift(df, baselines)
        assert report.feature_results[0].mean_shift == pytest.approx(2.0, abs=0.01)

    def test_out_of_range_computed(self):
        baselines = [FeatureBaseline(
            name="x", mean=0.0, std=1.0, min=-3.0, max=3.0,
            p5=-1.6, p25=-0.7, p50=0.0, p75=0.7, p95=1.6, n_samples=200,
        )]
        # Half the values are out of [-3, 3]
        df = pd.DataFrame({"x": np.concatenate([np.zeros(25), np.full(25, 10.0)])})
        report = detect_drift(df, baselines, min_samples=10)
        assert report.feature_results[0].out_of_range_pct == pytest.approx(0.5)

    def test_skips_features_with_few_samples(self):
        baselines = [FeatureBaseline(
            name="x", mean=0.0, std=1.0, min=-3.0, max=3.0,
            p5=-1.6, p25=-0.7, p50=0.0, p75=0.7, p95=1.6, n_samples=200,
        )]
        df = pd.DataFrame({"x": [1.0, 2.0]})  # Only 2 samples
        report = detect_drift(df, baselines, min_samples=30)
        assert report.n_features_checked == 0

    def test_report_summary(self):
        report = DriftReport(
            timestamp="2026-07-02", n_features_checked=10,
            n_drifted_warning=0, n_drifted_critical=0,
            overall_severity=DriftSeverity.NONE, feature_results=[],
        )
        assert "No drift" in report.summary()

    def test_missing_features_ignored(self):
        """Features in baseline but not in new data are skipped."""
        baselines = [FeatureBaseline(
            name="missing_col", mean=0.0, std=1.0, min=-3.0, max=3.0,
            p5=-1.6, p25=-0.7, p50=0.0, p75=0.7, p95=1.6, n_samples=200,
        )]
        df = pd.DataFrame({"other_col": np.zeros(50)})
        report = detect_drift(df, baselines)
        assert report.n_features_checked == 0
