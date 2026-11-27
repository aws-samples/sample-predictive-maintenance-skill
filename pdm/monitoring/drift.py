"""Detect distribution drift between reference (training) and new (inference) data.

Uses per-feature Kolmogorov-Smirnov tests against the reference distribution.
Reports missing and unusable sensor telemetry independently of distribution drift.
No external dependencies beyond scipy (already in core deps).
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats

from pdm.monitoring.baseline import FeatureBaseline


class DriftSeverity(Enum):
    NONE = "none"
    WARNING = "warning"       # p < 0.01
    CRITICAL = "critical"     # p < 0.001


class SensorHealthStatus(str, Enum):
    """Observed telemetry availability, not a diagnosis of hardware failure."""

    AVAILABLE = "available"
    MISSING_COLUMN = "missing_column"
    NO_VALID_SAMPLES = "no_valid_samples"
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    DEGRADED = "degraded"


@dataclass
class SensorHealthResult:
    """Availability of one expected numeric feature in the supplied window."""

    feature: str
    status: SensorHealthStatus
    n_rows: int
    n_valid: int
    n_missing: int  # absent column or null readings
    n_invalid: int  # non-numeric or non-finite readings

    @property
    def unusable_fraction(self) -> float | None:
        """Fraction of supplied rows without usable readings; None for no rows."""
        if self.n_rows == 0:
            return None
        return (self.n_missing + self.n_invalid) / self.n_rows

    @property
    def severity(self) -> DriftSeverity:
        if self.status in (SensorHealthStatus.MISSING_COLUMN,
                           SensorHealthStatus.NO_VALID_SAMPLES):
            return DriftSeverity.CRITICAL
        if self.status in (SensorHealthStatus.INSUFFICIENT_SAMPLES,
                           SensorHealthStatus.DEGRADED):
            return DriftSeverity.WARNING
        return DriftSeverity.NONE


@dataclass
class FeatureDriftResult:
    """Drift detection result for a single feature."""
    feature: str
    ks_statistic: float
    p_value: float
    severity: DriftSeverity
    mean_shift: float          # (new_mean - ref_mean) / ref_std
    out_of_range_pct: float    # fraction of samples outside [ref_min, ref_max]


@dataclass
class DriftReport:
    """Aggregate drift report across all features."""
    timestamp: str
    n_features_checked: int
    n_drifted_warning: int
    n_drifted_critical: int
    overall_severity: DriftSeverity
    feature_results: list[FeatureDriftResult]
    sensor_health_results: list[SensorHealthResult] = field(default_factory=list)

    @property
    def sensor_health_severity(self) -> DriftSeverity:
        severities = {r.severity for r in self.sensor_health_results}
        if DriftSeverity.CRITICAL in severities:
            return DriftSeverity.CRITICAL
        if DriftSeverity.WARNING in severities:
            return DriftSeverity.WARNING
        return DriftSeverity.NONE

    @property
    def is_healthy(self) -> bool:
        return self.overall_severity == DriftSeverity.NONE

    def summary(self) -> str:
        """Human-readable one-line summary."""
        if self.sensor_health_severity != DriftSeverity.NONE:
            critical = sum(r.severity == DriftSeverity.CRITICAL for r in self.sensor_health_results)
            warning = sum(r.severity == DriftSeverity.WARNING for r in self.sensor_health_results)
            return (f"⚠ Sensor data quality: {critical} critical, {warning} warning "
                    f"out of {len(self.sensor_health_results)} expected features; "
                    f"drift checks: {self.n_features_checked} tested "
                    f"({self.n_drifted_critical} critical, {self.n_drifted_warning} warning)")
        if self.is_healthy:
            return f"✓ No drift detected ({self.n_features_checked} features checked)"
        return (f"⚠ Drift detected: {self.n_drifted_critical} critical, "
                f"{self.n_drifted_warning} warning out of {self.n_features_checked} features")


def detect_drift(
    new_data: pd.DataFrame,
    baselines: list[FeatureBaseline],
    warning_threshold: float = 0.01,
    critical_threshold: float = 0.001,
    min_samples: int = 30,
    max_unusable_fraction: float = 0.10,
) -> DriftReport:
    """Detect distribution drift via KS test per feature.

    Compares finite numeric readings against stored baseline distributions.
    Sensor availability is reported for every baseline feature, including those
    that cannot be tested for drift. The overall severity combines the existing
    drift aggregation rule with sensor availability severity. Run this check
    before zero-filling or imputing missing telemetry.

    Args:
        new_data: New inference data to check
        baselines: Reference baselines from training (via capture_baseline)
        warning_threshold: p-value below which drift is flagged as warning
        critical_threshold: p-value below which drift is flagged as critical
        min_samples: Minimum finite numeric samples required to run drift test
        max_unusable_fraction: Warn if the fraction of supplied rows without a
            usable reading exceeds this value (default 0.10). Missing columns
            and channels with no usable readings are always critical; too few
            usable samples are always a warning.

    Returns:
        DriftReport with per-feature results and overall severity
    """
    if min_samples < 1:
        raise ValueError("min_samples must be positive")
    if not 0 <= max_unusable_fraction <= 1:
        raise ValueError("max_unusable_fraction must be between 0 and 1")
    if not new_data.columns.is_unique:
        raise ValueError("new_data must have unique feature columns")

    results: list[FeatureDriftResult] = []
    health_results: list[SensorHealthResult] = []
    n_rows = len(new_data)

    for bl in baselines:
        if bl.name not in new_data.columns:
            health_results.append(SensorHealthResult(
                bl.name, SensorHealthStatus.MISSING_COLUMN, n_rows, 0, n_rows, 0
            ))
            continue

        readings = new_data[bl.name]
        missing = readings.isna().to_numpy()
        # Accept numeric strings from telemetry ingestion; do not impute invalid
        # readings to zero (zero may be a legitimate voltage/current reading).
        numeric = pd.to_numeric(readings, errors="coerce")
        values = numeric.to_numpy(dtype=float, na_value=np.nan)
        valid = np.isfinite(values) & ~missing
        new_vals = values[valid]
        n_valid = len(new_vals)
        n_missing = int(missing.sum())
        n_invalid = n_rows - n_valid - n_missing

        if n_valid == 0:
            status = SensorHealthStatus.NO_VALID_SAMPLES
        elif n_valid < min_samples:
            status = SensorHealthStatus.INSUFFICIENT_SAMPLES
        elif (n_missing + n_invalid) / n_rows > max_unusable_fraction:
            status = SensorHealthStatus.DEGRADED
        else:
            status = SensorHealthStatus.AVAILABLE

        health_results.append(SensorHealthResult(
            bl.name, status, n_rows, n_valid, n_missing, n_invalid
        ))
        if n_valid < min_samples:
            continue

        # Generate reference sample from stored statistics (normal approximation)
        rng = np.random.default_rng(42)
        ref_sample = rng.normal(bl.mean, max(bl.std, 1e-10), bl.n_samples)

        # Two-sample KS test
        ks_stat, p_val = stats.ks_2samp(ref_sample, new_vals)

        # Determine severity
        if p_val < critical_threshold:
            severity = DriftSeverity.CRITICAL
        elif p_val < warning_threshold:
            severity = DriftSeverity.WARNING
        else:
            severity = DriftSeverity.NONE

        # Compute diagnostics
        mean_shift = (float(new_vals.mean()) - bl.mean) / max(bl.std, 1e-10)
        out_of_range = float(((new_vals < bl.min) | (new_vals > bl.max)).mean())

        results.append(FeatureDriftResult(
            feature=bl.name,
            ks_statistic=float(ks_stat),
            p_value=float(p_val),
            severity=severity,
            mean_shift=mean_shift,
            out_of_range_pct=out_of_range,
        ))

    n_warn = sum(1 for r in results if r.severity == DriftSeverity.WARNING)
    n_crit = sum(1 for r in results if r.severity == DriftSeverity.CRITICAL)

    if n_crit > 0:
        overall = DriftSeverity.CRITICAL
    elif n_warn > len(results) * 0.3:
        overall = DriftSeverity.WARNING
    else:
        overall = DriftSeverity.NONE

    # Preserve the historical KS aggregation above. Missing or degraded
    # telemetry can independently raise the overall monitoring severity.
    health_severities = {r.severity for r in health_results}
    if DriftSeverity.CRITICAL in health_severities:
        overall = DriftSeverity.CRITICAL
    elif DriftSeverity.WARNING in health_severities and overall == DriftSeverity.NONE:
        overall = DriftSeverity.WARNING

    return DriftReport(
        timestamp=datetime.utcnow().isoformat(),
        n_features_checked=len(results),
        n_drifted_warning=n_warn,
        n_drifted_critical=n_crit,
        overall_severity=overall,
        feature_results=sorted(results, key=lambda r: r.p_value),
        sensor_health_results=health_results,
    )
