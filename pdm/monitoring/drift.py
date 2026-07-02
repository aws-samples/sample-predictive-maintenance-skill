"""Detect distribution drift between reference (training) and new (inference) data.

Uses per-feature Kolmogorov-Smirnov tests against the reference distribution.
No external dependencies beyond scipy (already in core deps).
"""
from dataclasses import dataclass
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

    @property
    def is_healthy(self) -> bool:
        return self.overall_severity == DriftSeverity.NONE

    def summary(self) -> str:
        """Human-readable one-line summary."""
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
) -> DriftReport:
    """Detect distribution drift via KS test per feature.

    Compares new data against stored baseline distributions. Each feature
    is tested independently; the overall severity is the worst individual result.

    Args:
        new_data: New inference data to check
        baselines: Reference baselines from training (via capture_baseline)
        warning_threshold: p-value below which drift is flagged as warning
        critical_threshold: p-value below which drift is flagged as critical
        min_samples: Minimum samples required to run drift test

    Returns:
        DriftReport with per-feature results and overall severity
    """
    results: list[FeatureDriftResult] = []

    for bl in baselines:
        if bl.name not in new_data.columns:
            continue

        new_vals = new_data[bl.name].dropna().values
        if len(new_vals) < min_samples:
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

    return DriftReport(
        timestamp=datetime.utcnow().isoformat(),
        n_features_checked=len(results),
        n_drifted_warning=n_warn,
        n_drifted_critical=n_crit,
        overall_severity=overall,
        feature_results=sorted(results, key=lambda r: r.p_value),
    )
