"""Opt-in detection of suspected ADC saturation at configured measurement rails.

Rail occupancy is an observable symptom, not proof of analog clipping.
Samples beyond the configured range are reported separately, not counted as
clipped samples. Analyze one asset's ordered, unmodified telemetry at a time.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import math
from numbers import Integral, Real

import numpy as np
import pandas as pd


class SaturationStatus(str, Enum):
    MISSING_COLUMN = "missing_column"
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    NO_SATURATION_DETECTED = "no_saturation_detected"
    SUSPECTED_SATURATION = "suspected_saturation"


def _finite_real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    return number


@dataclass(frozen=True)
class SaturationConfig:
    """Measurement limits, in-range rail tolerance and minimum sustained run.

    Limits and tolerance are in the same units as the supplied samples.
    Rail bands include their boundaries and must not overlap.
    """

    lower_limit: float
    upper_limit: float
    min_consecutive_samples: int
    tolerance: float = 0.0
    rail_fraction_threshold: float | None = None

    def __post_init__(self):
        lower = _finite_real(self.lower_limit, "lower_limit")
        upper = _finite_real(self.upper_limit, "upper_limit")
        tolerance = _finite_real(self.tolerance, "tolerance")
        if lower >= upper:
            raise ValueError("lower_limit must be less than upper_limit")
        if tolerance < 0 or lower + tolerance >= upper - tolerance:
            raise ValueError("tolerance must be nonnegative and rail bands must not overlap")
        if (isinstance(self.min_consecutive_samples, (bool, np.bool_))
                or not isinstance(self.min_consecutive_samples, Integral)
                or self.min_consecutive_samples < 2):
            raise ValueError("min_consecutive_samples must be an integer >= 2")
        if self.rail_fraction_threshold is not None:
            fraction = _finite_real(self.rail_fraction_threshold, "rail_fraction_threshold")
            if not 0 < fraction <= 1:
                raise ValueError("rail_fraction_threshold must be in (0, 1]")
            object.__setattr__(self, "rail_fraction_threshold", fraction)
        object.__setattr__(self, "lower_limit", lower)
        object.__setattr__(self, "upper_limit", upper)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "min_consecutive_samples", int(self.min_consecutive_samples))


@dataclass(frozen=True)
class RailActivity:
    """Count and persistence of samples within one configured rail band."""

    n_samples: int
    longest_run: int
    trailing_run: int
    min_consecutive_samples: int

    @property
    def suspected(self) -> bool:
        return self.longest_run >= self.min_consecutive_samples

    @property
    def ongoing(self) -> bool:
        return self.trailing_run >= self.min_consecutive_samples


@dataclass(frozen=True)
class SaturationResult:
    feature: str
    status: SaturationStatus
    n_rows: int
    n_in_range: int
    n_invalid: int
    n_below_range: int
    n_above_range: int
    longest_in_range_run: int
    lower: RailActivity
    upper: RailActivity
    occupancy_suspected: bool = False

    @property
    def rail_fraction(self) -> float | None:
        """Share of in-range samples occupying either rail; None if none usable."""
        if not self.n_in_range:
            return None
        return (self.lower.n_samples + self.upper.n_samples) / self.n_in_range

    @property
    def ongoing(self) -> bool:
        return self.lower.ongoing or self.upper.ongoing


def _reading(value) -> float | None:
    # A timestamp, boolean, complex value or missing value is not a real
    # analog amplitude. Numeric strings from telemetry ingestion are accepted.
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, str)):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _check_channel(
    feature: str, readings: pd.Series, config: SaturationConfig
) -> SaturationResult:
    n_in_range = n_invalid = n_below = n_above = 0
    valid_run = longest_valid = 0
    counts = [0, 0]
    longest = [0, 0]
    trailing = [0, 0]
    lower_band = config.lower_limit + config.tolerance
    upper_band = config.upper_limit - config.tolerance

    for raw in readings:
        value = _reading(raw)
        rail = None
        if value is None:
            n_invalid += 1
            valid_run = 0
        elif value < config.lower_limit:
            n_below += 1
            valid_run = 0
        elif value > config.upper_limit:
            n_above += 1
            valid_run = 0
        else:
            n_in_range += 1
            valid_run += 1
            longest_valid = max(longest_valid, valid_run)
            if value <= lower_band:
                rail = 0
            elif value >= upper_band:
                rail = 1

        # A gap, interior sample, or opposite rail ends a same-rail run.
        for i in (0, 1):
            if rail == i:
                counts[i] += 1
                trailing[i] += 1
                longest[i] = max(longest[i], trailing[i])
            else:
                trailing[i] = 0

    required = config.min_consecutive_samples
    lower, upper = (
        RailActivity(counts[i], longest[i], trailing[i], required)
        for i in (0, 1)
    )
    # The optional aggregate criterion can detect repeated clipped peaks
    # even when no single rail run meets the consecutive-sample threshold.
    occupancy_suspected = (
        config.rail_fraction_threshold is not None
        and n_in_range >= required
        and (lower.n_samples + upper.n_samples) / n_in_range
        >= config.rail_fraction_threshold
    )
    if lower.suspected or upper.suspected or occupancy_suspected:
        status = SaturationStatus.SUSPECTED_SATURATION
    elif longest_valid < required:
        status = SaturationStatus.INSUFFICIENT_SAMPLES
    else:
        status = SaturationStatus.NO_SATURATION_DETECTED

    return SaturationResult(
        feature, status, len(readings), n_in_range, n_invalid, n_below,
        n_above, longest_valid, lower, upper, occupancy_suspected
    )


def detect_saturation(
    data: pd.DataFrame, configs: Mapping[str, SaturationConfig]
) -> list[SaturationResult]:
    """Report suspected sustained rail occupancy for explicitly configured sensors.

    Use sample order as supplied. Invalid or out-of-range samples break runs.
    By default, only sustained same-rail occupancy triggers suspicion. When a
    rail_fraction_threshold is configured, aggregate rail occupancy may also
    trigger suspicion with at least min_consecutive_samples in-range readings
    in the entire window. The aggregate criterion is not an ongoing run.
    No saturation detected does not establish telemetry or equipment health.
    """
    if not data.columns.is_unique:
        raise ValueError("data must have unique feature columns")

    results = []
    for feature, config in configs.items():
        if not isinstance(config, SaturationConfig):
            raise TypeError("configs must contain SaturationConfig values")
        if feature not in data.columns:
            empty = RailActivity(0, 0, 0, config.min_consecutive_samples)
            results.append(SaturationResult(
                feature, SaturationStatus.MISSING_COLUMN, len(data), 0, 0,
                0, 0, 0, empty, empty
            ))
            continue
        results.append(_check_channel(feature, data[feature], config))
    return results
