"""Tests for pdm/monitoring — baseline capture and drift detection."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from pdm.monitoring.baseline import capture_baseline, save_baseline, load_baseline, FeatureBaseline
from pdm.monitoring.drift import (
    detect_drift, DriftSeverity, DriftReport, SensorHealthStatus,
)


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

    def test_missing_features_not_tested_but_flagged(self):
        """A missing sensor is not a successful drift check."""
        baselines = [FeatureBaseline(
            name="missing_col", mean=0.0, std=1.0, min=-3.0, max=3.0,
            p5=-1.6, p25=-0.7, p50=0.0, p75=0.7, p95=1.6, n_samples=200,
        )]
        df = pd.DataFrame({"other_col": np.zeros(50)})
        report = detect_drift(df, baselines)
        assert report.n_features_checked == 0

        assert report.overall_severity == DriftSeverity.CRITICAL
        assert not report.is_healthy
        assert report.sensor_health_results[0].status == SensorHealthStatus.MISSING_COLUMN
        assert "Sensor data quality" in report.summary()


def _sensor_baselines(*names):
    return [
        FeatureBaseline(name=name, mean=0.0, std=1.0, min=-3.0, max=3.0,
                        p5=-1.6, p25=-0.7, p50=0.0, p75=0.7, p95=1.6, n_samples=200)
        for name in names
    ]


class TestSensorHealth:
    def test_missing_channel_does_not_hide_healthy_channel_or_report_healthy(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"vibration": rng.normal(size=100)})
        report = detect_drift(df, _sensor_baselines("motor_current", "vibration"))
        health = {item.feature: item for item in report.sensor_health_results}
        assert report.n_features_checked == 1
        assert report.overall_severity == DriftSeverity.CRITICAL
        assert report.sensor_health_severity == DriftSeverity.CRITICAL
        assert not report.is_healthy
        missing = health["motor_current"]
        assert missing.status == SensorHealthStatus.MISSING_COLUMN
        assert (missing.n_rows, missing.n_valid, missing.n_missing, missing.n_invalid) == (100, 0, 100, 0)
        assert missing.unusable_fraction == 1.0
        assert health["vibration"].status == SensorHealthStatus.AVAILABLE
        assert [r.feature for r in report.feature_results] == ["vibration"]
        assert "No drift detected" not in report.summary()

    @pytest.mark.parametrize("readings,missing,invalid", [
        ([None] * 50, 50, 0),
        ([np.nan] * 50, 50, 0),
        ([np.inf, -np.inf] * 25, 0, 50),
        (["bad"] * 50, 0, 50),
        ([None, np.inf, "bad", pd.NA] * 10, 20, 20),
    ])
    def test_channel_with_no_usable_readings(self, readings, missing, invalid):
        report = detect_drift(pd.DataFrame({"voltage": readings}), _sensor_baselines("voltage"))
        item = report.sensor_health_results[0]
        assert item.status == SensorHealthStatus.NO_VALID_SAMPLES
        assert item.n_valid == 0
        assert (item.n_missing, item.n_invalid) == (missing, invalid)
        assert item.unusable_fraction == 1.0
        assert report.n_features_checked == 0
        assert report.overall_severity == DriftSeverity.CRITICAL
        assert not report.is_healthy

    def test_empty_window_is_not_healthy(self):
        report = detect_drift(pd.DataFrame({"voltage": []}), _sensor_baselines("voltage"))
        item = report.sensor_health_results[0]
        assert item.status == SensorHealthStatus.NO_VALID_SAMPLES
        assert item.unusable_fraction is None
        assert report.n_features_checked == 0
        assert not report.is_healthy

    def test_insufficient_valid_samples_is_warning_not_a_drift_result(self):
        readings = [0.1, 0.0] + [None] * 48
        report = detect_drift(pd.DataFrame({"current": readings}), _sensor_baselines("current"))
        item = report.sensor_health_results[0]
        assert item.status == SensorHealthStatus.INSUFFICIENT_SAMPLES
        assert item.n_valid == 2
        assert report.n_features_checked == 0
        assert report.n_drifted_warning == report.n_drifted_critical == 0
        assert report.overall_severity == DriftSeverity.WARNING
        assert not report.is_healthy

    def test_valid_zero_and_numeric_strings_are_not_missing(self):
        readings = pd.Series(["0", "1.5", 0, -2.0] * 10, dtype=object)
        report = detect_drift(pd.DataFrame({"current": readings}), _sensor_baselines("current"))
        item = report.sensor_health_results[0]
        assert item.status == SensorHealthStatus.AVAILABLE
        assert item.n_valid == 40
        assert item.n_missing == item.n_invalid == 0
        assert item.unusable_fraction == 0.0
        assert report.n_features_checked == 1

    def test_degraded_sensor_still_tests_only_finite_values(self):
        rng = np.random.default_rng(42)
        values = rng.normal(size=80)
        readings = values.tolist() + [None] * 10 + [np.inf] * 5 + ["bad"] * 5
        report = detect_drift(pd.DataFrame({"voltage": readings}), _sensor_baselines("voltage"))
        item = report.sensor_health_results[0]
        assert item.status == SensorHealthStatus.DEGRADED
        assert (item.n_valid, item.n_missing, item.n_invalid) == (80, 10, 10)
        assert item.unusable_fraction == pytest.approx(0.20)
        assert report.sensor_health_severity == DriftSeverity.WARNING
        assert report.n_features_checked == 1
        clean = detect_drift(pd.DataFrame({"voltage": values}), _sensor_baselines("voltage"))
        assert report.feature_results[0] == clean.feature_results[0]
        assert not report.is_healthy

    def test_configurable_dropout_tolerance_and_boundary(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.normal(size=90).tolist() + [None] * 10})
        default = detect_drift(df, _sensor_baselines("x"))
        strict = detect_drift(df, _sensor_baselines("x"), max_unusable_fraction=0.05)
        assert default.sensor_health_results[0].status == SensorHealthStatus.AVAILABLE
        assert strict.sensor_health_results[0].status == SensorHealthStatus.DEGRADED

    def test_distribution_drift_remains_critical_despite_available_sensor(self):
        df = pd.DataFrame({"voltage": np.full(100, 10.0)})
        report = detect_drift(df, _sensor_baselines("voltage"))
        assert report.sensor_health_severity == DriftSeverity.NONE
        assert report.overall_severity == DriftSeverity.CRITICAL
        assert report.n_drifted_critical == 1

    def test_nullable_numeric_series(self):
        df = pd.DataFrame({"x": pd.Series([1.0] * 40 + [pd.NA], dtype="Float64")})
        item = detect_drift(df, _sensor_baselines("x")).sensor_health_results[0]
        assert (item.n_valid, item.n_missing, item.n_invalid) == (40, 1, 0)

    @pytest.mark.parametrize("kwargs", [
        {"min_samples": 0},
        {"max_unusable_fraction": -0.01},
        {"max_unusable_fraction": 1.01},
        {"max_unusable_fraction": np.nan},
    ])
    def test_invalid_monitoring_configuration(self, kwargs):
        with pytest.raises(ValueError):
            detect_drift(pd.DataFrame({"x": [0.0]}), _sensor_baselines("x"), **kwargs)
