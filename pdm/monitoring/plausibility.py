"""Opt-in cross-sensor checks using explicitly configured physical relationships.

Rows must contain synchronized measurements from one asset and a known
operating regime. A residual violation is evidence of inconsistency with
the configured relationship, not identification of the faulty sensor.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import math
from numbers import Integral, Real

import numpy as np
import pandas as pd


class Relation(str, Enum):
    DC_POWER = "dc_power"
    OHMS_LAW = "ohms_law"
    REDUNDANT = "redundant"


class PlausibilityStatus(str, Enum):
    MISSING_COLUMNS = "missing_columns"
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    NO_VIOLATION = "no_violation"
    SUSPECTED_INCONSISTENCY = "suspected_inconsistency"


def _finite_real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    return number


@dataclass(frozen=True)
class PlausibilityConfig:
    """Relationship, ordered input columns and residual tolerance.

    DC_POWER: measured power (W), inputs voltage (V), current (A).
    OHMS_LAW: measured voltage (V), inputs resistance (ohm), current (A).
    REDUNDANT: measured quantity, one independent reference in the same units.
    """

    relation: Relation
    measured: str
    inputs: tuple[str, ...]
    absolute_tolerance: float
    relative_tolerance: float
    min_consecutive_samples: int

    def __post_init__(self):
        if not isinstance(self.relation, Relation):
            raise ValueError("relation must be a Relation")
        required = 1 if self.relation == Relation.REDUNDANT else 2
        if not isinstance(self.inputs, tuple) or len(self.inputs) != required:
            raise ValueError(f"relation requires {required} input column(s)")
        columns = (self.measured, *self.inputs)
        if any(not isinstance(col, str) or not col.strip() for col in columns):
            raise ValueError("column names must be nonempty strings")
        if len(set(columns)) != len(columns):
            raise ValueError("measured and input columns must be distinct")
        for name in ("absolute_tolerance", "relative_tolerance"):
            value = _finite_real(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        minimum = self.min_consecutive_samples
        if (isinstance(minimum, (bool, np.bool_))
                or not isinstance(minimum, Integral) or minimum < 2):
            raise ValueError("min_consecutive_samples must be an integer >= 2")
        object.__setattr__(self, "min_consecutive_samples", int(minimum))


@dataclass(frozen=True)
class PlausibilityResult:
    rule: str
    relation: Relation
    status: PlausibilityStatus
    missing_columns: tuple[str, ...]
    n_rows: int
    n_evaluated: int
    n_unusable: int
    n_violations: int
    longest_evaluated_run: int
    longest_violation_run: int
    trailing_violation_run: int
    min_consecutive_samples: int
    max_absolute_residual: float | None

    @property
    def violation_fraction(self) -> float | None:
        if not self.n_evaluated:
            return None
        return self.n_violations / self.n_evaluated

    @property
    def ongoing(self) -> bool:
        return self.trailing_violation_run >= self.min_consecutive_samples


def _reading(value) -> float | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, str)):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _check_rule(
    name: str, data: pd.DataFrame, config: PlausibilityConfig
) -> PlausibilityResult:
    columns = (config.measured, *config.inputs)
    missing = tuple(col for col in columns if col not in data.columns)
    required = config.min_consecutive_samples
    if missing:
        return PlausibilityResult(
            name, config.relation, PlausibilityStatus.MISSING_COLUMNS, missing,
            len(data), 0, 0, 0, 0, 0, 0, required, None
        )

    evaluated = unusable = violations = 0
    evaluated_run = longest_evaluated = run = longest = 0
    max_residual = None
    for row in data.loc[:, list(columns)].itertuples(index=False, name=None):
        values = [_reading(value) for value in row]
        # Negative resistance is outside this passive-resistor rule's domain.
        if any(value is None for value in values) or (
            config.relation == Relation.OHMS_LAW and values[1] < 0
        ):
            unusable += 1
            evaluated_run = run = 0
            continue

        measured = values[0]
        expected = values[1] if config.relation == Relation.REDUNDANT else values[1] * values[2]
        residual = abs(measured - expected)
        # Relative tolerance is anchored to the expected/reference magnitude;
        # the absolute tolerance handles expected values near zero.
        tolerance = config.absolute_tolerance + config.relative_tolerance * abs(expected)
        if not all(math.isfinite(value) for value in (expected, residual, tolerance)):
            unusable += 1
            evaluated_run = run = 0
            continue

        evaluated += 1
        evaluated_run += 1
        longest_evaluated = max(longest_evaluated, evaluated_run)
        max_residual = residual if max_residual is None else max(max_residual, residual)
        if residual > tolerance:
            violations += 1
            run += 1
            longest = max(longest, run)
        else:
            run = 0

    if longest >= required:
        status = PlausibilityStatus.SUSPECTED_INCONSISTENCY
    elif longest_evaluated < required:
        status = PlausibilityStatus.INSUFFICIENT_SAMPLES
    else:
        status = PlausibilityStatus.NO_VIOLATION

    return PlausibilityResult(
        name, config.relation, status, (), len(data), evaluated, unusable,
        violations, longest_evaluated, longest, run, required, max_residual
    )


def check_plausibility(
    data: pd.DataFrame, rules: Mapping[str, PlausibilityConfig]
) -> list[PlausibilityResult]:
    """Check explicit relationships in supplied row order, without imputation.

    A violation requires abs(measured - expected) greater than the sum of
    absolute tolerance and relative tolerance times abs(expected). Invalid
    or nonfinite computations interrupt consecutive runs. A no-violation
    status means no sustained qualifying violation, not proof of health.
    """
    if not data.columns.is_unique:
        raise ValueError("data must have unique feature columns")
    results = []
    for name, config in rules.items():
        if not isinstance(config, PlausibilityConfig):
            raise TypeError("rules must contain PlausibilityConfig values")
        results.append(_check_rule(name, data, config))
    return results
