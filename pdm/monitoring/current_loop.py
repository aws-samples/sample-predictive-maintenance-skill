"""Opt-in diagnostics for unipolar industrial current loops, in milliamperes.

Classifies observed signal bands, not the physical cause of a fault. Thresholds
must match the actual transmitter and receiver configuration.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import math
from numbers import Integral, Real

import numpy as np
import pandas as pd


class CurrentBand(str, Enum):
    INVALID = "invalid"
    LOW_ALARM = "low_alarm"
    LOW_TRANSITION = "low_transition"
    UNDERRANGE = "underrange"
    NOMINAL = "nominal"
    OVERRANGE = "overrange"
    HIGH_TRANSITION = "high_transition"
    HIGH_ALARM = "high_alarm"


class CurrentLoopStatus(str, Enum):
    MISSING_COLUMN = "missing_column"
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    NO_SUSTAINED_ALARM = "no_sustained_alarm"
    SUSPECTED_ALARM = "suspected_alarm"


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
class CurrentLoopConfig:
    """Explicitly ordered unipolar current thresholds and alarm persistence."""

    low_alarm_ma: float
    low_measurement_ma: float
    nominal_min_ma: float
    nominal_max_ma: float
    high_measurement_ma: float
    high_alarm_ma: float
    min_consecutive_samples: int

    def __post_init__(self):
        names = (
            "low_alarm_ma", "low_measurement_ma", "nominal_min_ma",
            "nominal_max_ma", "high_measurement_ma", "high_alarm_ma",
        )
        values = [_finite_real(getattr(self, name), name) for name in names]
        if values[0] < 0 or not all(a < b for a, b in zip(values, values[1:])):
            raise ValueError("current thresholds must be nonnegative and strictly increasing")
        minimum = self.min_consecutive_samples
        if (isinstance(minimum, (bool, np.bool_))
                or not isinstance(minimum, Integral) or minimum < 2):
            raise ValueError("min_consecutive_samples must be an integer >= 2")
        for name, value in zip(names, values):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "min_consecutive_samples", int(minimum))

    @classmethod
    def ne43_example(cls, min_consecutive_samples: int) -> "CurrentLoopConfig":
        """Illustrative NE43-style thresholds; verify actual device settings."""
        return cls(3.6, 3.8, 4.0, 20.0, 20.5, 21.0, min_consecutive_samples)


def classify_current(value, config: CurrentLoopConfig) -> CurrentBand:
    """Classify one current reading in mA, with explicit boundary semantics.

    Alarm boundaries are inclusive. Measurement endpoints are included in
    the underrange/overrange bands. Nominal endpoints are inclusive.
    Negative and unusable readings are invalid for this unipolar profile.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, str)):
        return CurrentBand.INVALID
    try:
        current = float(value)
    except (ValueError, OverflowError):
        return CurrentBand.INVALID
    if not math.isfinite(current) or current < 0:
        return CurrentBand.INVALID
    if current <= config.low_alarm_ma:
        return CurrentBand.LOW_ALARM
    if current < config.low_measurement_ma:
        return CurrentBand.LOW_TRANSITION
    if current < config.nominal_min_ma:
        return CurrentBand.UNDERRANGE
    if current <= config.nominal_max_ma:
        return CurrentBand.NOMINAL
    if current <= config.high_measurement_ma:
        return CurrentBand.OVERRANGE
    if current < config.high_alarm_ma:
        return CurrentBand.HIGH_TRANSITION
    return CurrentBand.HIGH_ALARM


@dataclass(frozen=True)
class BandActivity:
    n_samples: int
    longest_run: int
    trailing_run: int


@dataclass(frozen=True)
class CurrentLoopResult:
    feature: str
    status: CurrentLoopStatus
    n_rows: int
    n_valid: int
    longest_valid_run: int
    min_consecutive_samples: int
    bands: dict[CurrentBand, BandActivity]

    @property
    def n_measurement(self) -> int:
        """Samples in nominal or permitted extended measurement bands."""
        return sum(self.bands[b].n_samples for b in (
            CurrentBand.UNDERRANGE, CurrentBand.NOMINAL, CurrentBand.OVERRANGE
        ))

    @property
    def low_alarm_suspected(self) -> bool:
        return self.bands[CurrentBand.LOW_ALARM].longest_run >= self.min_consecutive_samples

    @property
    def high_alarm_suspected(self) -> bool:
        return self.bands[CurrentBand.HIGH_ALARM].longest_run >= self.min_consecutive_samples

    @property
    def ongoing(self) -> bool:
        return any(self.bands[b].trailing_run >= self.min_consecutive_samples
                   for b in (CurrentBand.LOW_ALARM, CurrentBand.HIGH_ALARM))


def _empty_bands() -> dict[CurrentBand, BandActivity]:
    return {band: BandActivity(0, 0, 0) for band in CurrentBand}


def _check_channel(feature: str, readings: pd.Series,
                   config: CurrentLoopConfig) -> CurrentLoopResult:
    counts = dict.fromkeys(CurrentBand, 0)
    longest = dict.fromkeys(CurrentBand, 0)
    n_valid = valid_run = longest_valid = run = 0
    previous = None
    for reading in readings:
        band = classify_current(reading, config)
        counts[band] += 1
        run = run + 1 if band == previous else 1
        longest[band] = max(longest[band], run)
        previous = band
        if band == CurrentBand.INVALID:
            valid_run = 0
        else:
            n_valid += 1
            valid_run += 1
            longest_valid = max(longest_valid, valid_run)

    bands = {band: BandActivity(counts[band], longest[band],
                               run if band == previous else 0)
             for band in CurrentBand}
    required = config.min_consecutive_samples
    if any(bands[b].longest_run >= required
           for b in (CurrentBand.LOW_ALARM, CurrentBand.HIGH_ALARM)):
        status = CurrentLoopStatus.SUSPECTED_ALARM
    elif longest_valid < required:
        status = CurrentLoopStatus.INSUFFICIENT_SAMPLES
    else:
        status = CurrentLoopStatus.NO_SUSTAINED_ALARM
    return CurrentLoopResult(feature, status, len(readings), n_valid,
                             longest_valid, required, bands)


def detect_current_loop(
    data: pd.DataFrame, configs: Mapping[str, CurrentLoopConfig]
) -> list[CurrentLoopResult]:
    """Assess configured channels in supplied sample order, in milliamperes.

    Transition, measurement, invalid and opposite alarm samples interrupt
    same-direction alarm runs. All bands remain visible even when no sustained
    alarm is detected. This is not a hardware diagnosis or a safety interlock.
    """
    if not data.columns.is_unique:
        raise ValueError("data must have unique feature columns")
    results = []
    for feature, config in configs.items():
        if not isinstance(config, CurrentLoopConfig):
            raise TypeError("configs must contain CurrentLoopConfig values")
        if feature not in data.columns:
            results.append(CurrentLoopResult(
                feature, CurrentLoopStatus.MISSING_COLUMN, len(data), 0, 0,
                config.min_consecutive_samples, _empty_bands()
            ))
        else:
            results.append(_check_channel(feature, data[feature], config))
    return results
