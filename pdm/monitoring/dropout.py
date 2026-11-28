"""Opt-in intermittent and sustained sensor dropout diagnostics.

Analyze supplied sample rows, before imputation, on one asset and one
continuous ordered acquisition session. Absence of entire timestamp rows
cannot be inferred from this utility; use explicit missing observations.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import math
from numbers import Integral, Real

import numpy as np
import pandas as pd


class DropoutStatus(str, Enum):
    MISSING_COLUMN = "missing_column"
    EMPTY_WINDOW = "empty_window"
    NO_VALID_SAMPLES = "no_valid_samples"
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    NO_QUALIFYING_DROPOUT = "no_qualifying_dropout"
    SUSPECTED_INTERMITTENT = "suspected_intermittent"
    SUSPECTED_SUSTAINED = "suspected_sustained"
    SUSPECTED_BOTH = "suspected_both"


@dataclass(frozen=True)
class DropoutConfig:
    """Explicit sample-count thresholds, not time durations."""

    min_recovered_episodes: int
    sustained_run_samples: int

    def __post_init__(self):
        for name in ("min_recovered_episodes", "sustained_run_samples"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, Integral) or value < 2):
                raise ValueError(f"{name} must be an integer >= 2")
            object.__setattr__(self, name, int(value))


@dataclass(frozen=True)
class DropoutResult:
    feature: str
    status: DropoutStatus
    n_rows: int
    n_valid: int
    n_missing: int
    n_invalid: int
    n_episodes: int
    n_recovered_short_episodes: int
    n_sustained_episodes: int
    longest_dropout_run: int
    leading_dropout_run: int
    trailing_dropout_run: int
    config: DropoutConfig

    @property
    def dropout_fraction(self) -> float | None:
        """Unusable share of observed rows; undefined for absent/empty channels."""
        if self.status == DropoutStatus.MISSING_COLUMN or not self.n_rows:
            return None
        return (self.n_missing + self.n_invalid) / self.n_rows

    @property
    def intermittent_suspected(self) -> bool:
        return self.n_recovered_short_episodes >= self.config.min_recovered_episodes

    @property
    def sustained_suspected(self) -> bool:
        return self.n_sustained_episodes > 0

    @property
    def ongoing_dropout(self) -> bool:
        """A present channel ends with at least one unusable observation."""
        return self.trailing_dropout_run > 0

    @property
    def ongoing_sustained(self) -> bool:
        return self.trailing_dropout_run >= self.config.sustained_run_samples


def _is_finite_reading(value) -> bool:
    # Do not convert timestamps, booleans or complex values to amplitudes.
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, str)):
        return False
    try:
        return math.isfinite(float(value))
    except (ValueError, OverflowError):
        return False


def _check_channel(
    feature: str, readings: pd.Series, config: DropoutConfig
) -> DropoutResult:
    n_rows = len(readings)
    n_valid = n_missing = n_invalid = 0
    episodes = short_recovered = sustained = 0
    longest = leading = run = 0
    seen_valid = False

    # A null and an invalid amplitude both represent an unusable observation;
    # transitions between them do not artificially split a dropout episode.
    missing_mask = readings.isna().to_numpy()
    for missing, value in zip(missing_mask, readings):
        if missing or not _is_finite_reading(value):
            if missing:
                n_missing += 1
            else:
                n_invalid += 1
            run += 1
            longest = max(longest, run)
            continue

        n_valid += 1
        if run:
            episodes += 1
            if not seen_valid:
                leading = run
            if run >= config.sustained_run_samples:
                sustained += 1
            elif seen_valid:
                # Count an intermittent episode only if both boundaries are
                # observed. A leading gap may have begun before this window.
                short_recovered += 1
            run = 0
        seen_valid = True

    trailing = run
    if run:
        episodes += 1
        if not seen_valid:
            leading = run
        if run >= config.sustained_run_samples:
            sustained += 1

    intermittent = short_recovered >= config.min_recovered_episodes
    if not n_rows:
        status = DropoutStatus.EMPTY_WINDOW
    elif not n_valid:
        status = DropoutStatus.NO_VALID_SAMPLES
    elif intermittent and sustained:
        status = DropoutStatus.SUSPECTED_BOTH
    elif intermittent:
        status = DropoutStatus.SUSPECTED_INTERMITTENT
    elif sustained:
        status = DropoutStatus.SUSPECTED_SUSTAINED
    elif n_rows < min(config.sustained_run_samples,
                      2 * config.min_recovered_episodes + 1):
        # No qualifying pattern could fit in this window even in principle.
        status = DropoutStatus.INSUFFICIENT_SAMPLES
    else:
        status = DropoutStatus.NO_QUALIFYING_DROPOUT

    return DropoutResult(
        feature, status, n_rows, n_valid, n_missing, n_invalid, episodes,
        short_recovered, sustained, longest, leading, trailing, config
    )


def detect_dropouts(
    data: pd.DataFrame, configs: Mapping[str, DropoutConfig]
) -> list[DropoutResult]:
    """Detect recurrent bounded short gaps and sustained unusable-sample runs.

    Missing columns are distinct from present channels with unusable data.
    Zero, negative real values and finite numeric strings are valid readings.
    An all-unusable window is reported explicitly, even if it meets the
    sustained threshold. Diagnostic flags remain available in that case.
    No qualifying dropout does not establish equipment or sensor health.
    """
    if not data.columns.is_unique:
        raise ValueError("data must have unique feature columns")
    results = []
    for feature, config in configs.items():
        if not isinstance(config, DropoutConfig):
            raise TypeError("configs must contain DropoutConfig values")
        if feature not in data.columns:
            results.append(DropoutResult(
                feature, DropoutStatus.MISSING_COLUMN, len(data),
                0, 0, 0, 0, 0, 0, 0, 0, 0, config
            ))
        else:
            results.append(_check_channel(feature, data[feature], config))
    return results
