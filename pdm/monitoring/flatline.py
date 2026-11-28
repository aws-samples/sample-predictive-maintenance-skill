"""Opt-in detection of suspected flatlines in ordered sensor measurements.

This measures consecutive numeric samples within a configured signal range.
It does not establish a physical sensor fault or infer sampling intervals.
"""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import math
from numbers import Integral, Real

import numpy as np
import pandas as pd


class FlatlineStatus(str, Enum):
    MISSING_COLUMN = "missing_column"
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    NO_FLATLINE_DETECTED = "no_flatline_detected"
    SUSPECTED_FLATLINE = "suspected_flatline"


@dataclass(frozen=True)
class FlatlineConfig:
    """Maximum peak-to-peak range in sensor units and minimum run length."""

    tolerance: float
    min_consecutive_samples: int

    def __post_init__(self):
        if (isinstance(self.tolerance, bool) or not isinstance(self.tolerance, Real)
                or not math.isfinite(self.tolerance) or self.tolerance < 0):
            raise ValueError("tolerance must be a finite nonnegative number")
        if (isinstance(self.min_consecutive_samples, bool)
                or not isinstance(self.min_consecutive_samples, Integral)
                or self.min_consecutive_samples < 2):
            raise ValueError("min_consecutive_samples must be an integer >= 2")


@dataclass(frozen=True)
class FlatlineResult:
    feature: str
    status: FlatlineStatus
    n_rows: int
    n_valid: int
    longest_valid_run: int
    longest_flatline_run: int
    trailing_flatline_run: int

    # A recovered flatline may be historical rather than ongoing.
    ongoing: bool


def _run_lengths(values: np.ndarray, tolerance: float) -> tuple[int, int, int]:
    """Return longest valid, longest flatline, and trailing flatline run lengths.

    Monotonic queues track the minimum and maximum of a sliding window in
    linear time. Invalid samples reset the window. Checking the entire range
    (rather than successive differences) prevents slow ramps from looking flat.
    """
    minima = deque()
    maxima = deque()
    left = valid_run = longest_valid = longest_flatline = trailing = 0

    for right, raw in enumerate(values):
        value = float(raw)
        if not math.isfinite(value):
            minima.clear()
            maxima.clear()
            left = right + 1
            valid_run = trailing = 0
            continue

        valid_run += 1
        longest_valid = max(longest_valid, valid_run)
        while minima and float(values[minima[-1]]) >= value:
            minima.pop()
        while maxima and float(values[maxima[-1]]) <= value:
            maxima.pop()
        minima.append(right)
        maxima.append(right)

        while float(values[maxima[0]]) - float(values[minima[0]]) > tolerance:
            left += 1
            if minima[0] < left:
                minima.popleft()
            if maxima[0] < left:
                maxima.popleft()

        trailing = right - left + 1
        longest_flatline = max(longest_flatline, trailing)

    return longest_valid, longest_flatline, trailing


def detect_flatlines(
    data: pd.DataFrame, configs: Mapping[str, FlatlineConfig]
) -> list[FlatlineResult]:
    """Check explicitly configured sensors in the supplied row order.

    Use one asset and one ordered sampling window at a time, before imputation
    or zero-fill. Non-finite, null, or non-numeric readings break consecutive
    runs. No universal tolerance, sampling rate, or duration is assumed.
    An alert describes a candidate flatline, not a confirmed hardware fault.
    """
    if not data.columns.is_unique:
        raise ValueError("data must have unique feature columns")

    results = []
    for feature, config in configs.items():
        if not isinstance(config, FlatlineConfig):
            raise TypeError("configs must contain FlatlineConfig values")

        if feature not in data.columns:
            results.append(FlatlineResult(
                feature, FlatlineStatus.MISSING_COLUMN, len(data), 0, 0, 0, 0, False
            ))
            continue

        readings = data[feature]
        # Datetime, timedelta and complex signals are not real-valued sensor
        # amplitudes. Avoid silently treating their internal codes as readings.
        if (pd.api.types.is_datetime64_any_dtype(readings.dtype)
                or pd.api.types.is_timedelta64_dtype(readings.dtype)):
            values = np.full(len(readings), np.nan)
        else:
            numeric = pd.to_numeric(readings, errors="coerce")
            if pd.api.types.is_complex_dtype(numeric.dtype):
                values = np.full(len(readings), np.nan)
            else:
                values = numeric.to_numpy(dtype=float, na_value=np.nan)

        n_valid = int(np.isfinite(values).sum())
        longest_valid, longest_flatline, trailing = _run_lengths(
            values, config.tolerance
        )
        required = config.min_consecutive_samples
        if longest_valid < required:
            status = FlatlineStatus.INSUFFICIENT_SAMPLES
        elif longest_flatline >= required:
            status = FlatlineStatus.SUSPECTED_FLATLINE
        else:
            status = FlatlineStatus.NO_FLATLINE_DETECTED

        results.append(FlatlineResult(
            feature, status, len(data), n_valid, longest_valid,
            longest_flatline, trailing, trailing >= required
        ))
    return results
