"""Tests of physical cross-sensor consistency rules."""

import itertools

import numpy as np
import pandas as pd
import pytest

from pdm.monitoring.plausibility import (
    PlausibilityConfig, PlausibilityStatus, Relation, check_plausibility,
)


POWER = PlausibilityConfig(Relation.DC_POWER, "p", ("v", "i"), 1, 0.05, 3)
OHMS = PlausibilityConfig(Relation.OHMS_LAW, "v", ("r", "i"), 0.25, 0, 3)
REDUNDANT = PlausibilityConfig(Relation.REDUNDANT, "a", ("b",), 0.5, 0, 3)


def check(frame, config=POWER):
    return check_plausibility(pd.DataFrame(frame), {"rule": config})[0]


def test_dc_power_consistent_signed_and_zero_values():
    result = check({"v": [24, 24, 0, -24], "i": [2, -2, 3, 2],
                    "p": [48, -48, 0, -48]})
    assert result.status == PlausibilityStatus.NO_VIOLATION
    assert result.n_evaluated == 4
    assert result.max_absolute_residual == 0
    assert result.violation_fraction == 0


def test_dc_power_persistent_mismatch_and_recovery():
    result = check({"v": [24]*5, "i": [2]*5, "p": [48, 60, 60, 60, 48]})
    assert result.status == PlausibilityStatus.SUSPECTED_INCONSISTENCY
    assert result.longest_violation_run == 3
    assert result.n_violations == 3
    assert result.max_absolute_residual == 12
    assert not result.ongoing


def test_ohms_law_and_redundant_channels():
    result = check({"v": [12]*4, "r": [6]*4, "i": [2]*4}, OHMS)
    assert result.status == PlausibilityStatus.NO_VIOLATION
    result = check({"v": [14]*4, "r": [6]*4, "i": [2]*4}, OHMS)
    assert result.ongoing
    result = check({"a": [10, 11, 11, 11], "b": [10]*4}, REDUNDANT)
    assert result.ongoing


def test_combined_tolerance_inclusive_boundary_and_zero_reference():
    config = PlausibilityConfig(Relation.REDUNDANT, "a", ("b",), 1, 0.125, 2)
    result = check({"a": [10, -10, 1, -1], "b": [8, -8, 0, 0]}, config)
    assert result.n_violations == 0
    result = check({"a": [10.01, -10.01], "b": [8, -8]}, config)
    assert result.n_violations == 2 and result.ongoing


@pytest.mark.parametrize("gap", [None, np.nan, np.inf, "bad", True, 1+2j,
                                  pd.Timestamp("2026-01-01"), pd.NA])
def test_invalid_reading_breaks_consecutive_run(gap):
    frame = pd.DataFrame({"a": [2, 2, gap, 2, 2], "b": [0]*5}, dtype=object)
    result = check(frame, REDUNDANT)
    assert result.n_unusable == 1
    assert result.n_evaluated == 4
    assert result.longest_violation_run == 2
    assert result.status == PlausibilityStatus.INSUFFICIENT_SAMPLES
    assert not result.ongoing


def test_missing_columns_and_empty_window():
    result = check({"v": [24], "p": [48]})
    assert result.status == PlausibilityStatus.MISSING_COLUMNS
    assert result.missing_columns == ("i",)
    assert result.n_rows == 1 and result.n_evaluated == 0
    assert result.violation_fraction is None
    result = check({"p": [], "v": [], "i": []})
    assert result.status == PlausibilityStatus.INSUFFICIENT_SAMPLES
    assert result.max_absolute_residual is None


def test_negative_resistance_out_of_domain():
    result = check({"v": [12]*4, "r": [-6]*4, "i": [-2]*4}, OHMS)
    assert result.n_unusable == 4
    assert result.n_evaluated == 0
    assert result.status == PlausibilityStatus.INSUFFICIENT_SAMPLES


def test_nonfinite_derived_values_are_unusable():
    result = check({"v": [1e308]*4, "i": [1e308]*4, "p": [1]*4})
    assert result.n_unusable == 4
    assert result.max_absolute_residual is None
    result = check({"a": [1e308]*4, "b": [-1e308]*4}, REDUNDANT)
    assert result.n_unusable == 4
    config = PlausibilityConfig(Relation.REDUNDANT, "a", ("b",), 1, 10, 2)
    result = check({"a": [1e308]*2, "b": [1e308]*2}, config)
    assert result.n_unusable == 2


def test_numeric_strings_nullable_and_input_order():
    data = pd.DataFrame({"a": pd.array([2, 2, 2, pd.NA], dtype="Float64"),
                         "b": ["0"]*4}, index=[9, 3, 8, 1])
    result = check(data, REDUNDANT)
    assert result.n_evaluated == 3 and result.n_unusable == 1
    assert result.status == PlausibilityStatus.SUSPECTED_INCONSISTENCY
    assert not result.ongoing


def test_rule_independence_and_no_configuration():
    data = pd.DataFrame({"p": [60]*4, "v": [24]*4, "i": [2]*4,
                         "a": [5]*4, "b": [5]*4})
    results = check_plausibility(data, {"power": POWER, "redundant": REDUNDANT})
    assert [r.rule for r in results] == ["power", "redundant"]
    assert results[0].ongoing and not results[1].ongoing
    assert check_plausibility(data, {}) == []


@pytest.mark.parametrize("args", [
    ("dc_power", "p", ("v", "i"), 1, 0, 3),
    (Relation.DC_POWER, "p", ("v",), 1, 0, 3),
    (Relation.DC_POWER, "p", ("p", "i"), 1, 0, 3),
    (Relation.DC_POWER, "p", ("i", "i"), 1, 0, 3),
    (Relation.REDUNDANT, "", ("b",), 1, 0, 3),
    (Relation.REDUNDANT, "a", (" ",), 1, 0, 3),
    (Relation.REDUNDANT, "a", ["b"], 1, 0, 3),
    (Relation.REDUNDANT, "a", ("b",), -1, 0, 3),
    (Relation.REDUNDANT, "a", ("b",), 1, -0.1, 3),
    (Relation.REDUNDANT, "a", ("b",), np.inf, 0, 3),
    (Relation.REDUNDANT, "a", ("b",), 1, np.nan, 3),
    (Relation.REDUNDANT, "a", ("b",), True, 0, 3),
    (Relation.REDUNDANT, "a", ("b",), 1, 0, 1),
    (Relation.REDUNDANT, "a", ("b",), 1, 0, 2.5),
    (Relation.REDUNDANT, "a", ("b",), 1, 0, True),
])
def test_invalid_configuration(args):
    with pytest.raises(ValueError):
        PlausibilityConfig(*args)


def test_duplicate_columns_and_wrong_configuration():
    with pytest.raises(ValueError, match="unique"):
        check_plausibility(pd.DataFrame([[1, 2]], columns=["a", "a"]), {"r": REDUNDANT})
    with pytest.raises(TypeError, match="PlausibilityConfig"):
        check_plausibility(pd.DataFrame({"a": [1]}), {"r": {}})


def test_scan_against_independent_run_reference():
    for values in itertools.product([0.0, 2.0, None], repeat=5):
        result = check({"a": values, "b": [0]*5}, REDUNDANT)
        flags = [value == 2.0 for value in values]
        runs = [len(list(group)) for key, group in itertools.groupby(flags) if key]
        trailing = 0
        for flag in reversed(flags):
            if not flag:
                break
            trailing += 1
        assert result.n_violations == sum(flags)
        assert result.longest_violation_run == max(runs, default=0)
        assert result.trailing_violation_run == trailing
        assert result.n_evaluated + result.n_unusable == len(values)
