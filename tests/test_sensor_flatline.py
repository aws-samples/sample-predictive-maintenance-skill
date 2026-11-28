"""Industrial sensor flatline regression tests."""

import itertools

import numpy as np
import pandas as pd
import pytest

from pdm.monitoring.flatline import (
    FlatlineConfig, FlatlineStatus, detect_flatlines,
)


def check(values, tolerance=0.0, minimum=4):
    frame = pd.DataFrame({"motor_current": values})
    return detect_flatlines(
        frame, {"motor_current": FlatlineConfig(tolerance, minimum)}
    )[0]


def test_exact_stuck_at_and_valid_zero():
    for value in (12.0, 0.0, -1.5):
        result = check([value] * 6)
        assert result.status == FlatlineStatus.SUSPECTED_FLATLINE
        assert result.n_valid == 6
        assert result.longest_flatline_run == result.trailing_flatline_run == 6
        assert result.ongoing


def test_near_flatline_in_configured_sensor_units():
    values = [12.0, 12.125, 12.0, 12.0625]
    result = check(values, tolerance=0.125)
    assert result.status == FlatlineStatus.SUSPECTED_FLATLINE
    assert result.longest_flatline_run == 4
    assert check(values, tolerance=0.0625).status == FlatlineStatus.NO_FLATLINE_DETECTED


def test_slow_ramp_does_not_trigger_despite_small_adjacent_steps():
    result = check(np.arange(8) * 0.125, tolerance=0.125)
    assert result.status == FlatlineStatus.NO_FLATLINE_DETECTED
    assert result.longest_flatline_run == 2


@pytest.mark.parametrize("gap", [None, np.nan, np.inf, -np.inf, "bad", pd.NA])
def test_invalid_sample_breaks_consecutive_run(gap):
    result = check([2.0, 2.0, 2.0, gap, 2.0, 2.0, 2.0])
    assert result.n_valid == 6
    assert result.longest_valid_run == 3
    assert result.longest_flatline_run == 3
    assert result.status == FlatlineStatus.INSUFFICIENT_SAMPLES
    assert not result.ongoing


def test_historical_flatline_not_reported_as_ongoing_after_recovery():
    result = check([1.0] * 5 + [2.0, 3.0])
    assert result.status == FlatlineStatus.SUSPECTED_FLATLINE
    assert result.longest_flatline_run == 5
    assert result.trailing_flatline_run == 1
    assert not result.ongoing


def test_missing_final_sample_ends_ongoing_run():
    result = check([1.0] * 5 + [None])
    assert result.status == FlatlineStatus.SUSPECTED_FLATLINE
    assert result.trailing_flatline_run == 0
    assert not result.ongoing


def test_flatline_ending_at_last_row():
    result = check([1, 2, 3, 4, 4, 4, 4])
    assert result.longest_flatline_run == 4
    assert result.trailing_flatline_run == 4
    assert result.ongoing


def test_unconfigured_idle_channel_not_checked():
    data = pd.DataFrame({"idle_motor": [0] * 6, "voltage": range(6)})
    results = detect_flatlines(data, {"voltage": FlatlineConfig(0, 4)})
    assert [r.feature for r in results] == ["voltage"]
    assert results[0].status == FlatlineStatus.NO_FLATLINE_DETECTED
    assert detect_flatlines(data, {}) == []


def test_missing_column_distinct_from_empty_or_unusable_window():
    config = {"motor_current": FlatlineConfig(0, 4)}
    missing = detect_flatlines(pd.DataFrame({"other": [0]}), config)[0]
    assert missing.status == FlatlineStatus.MISSING_COLUMN
    assert missing.n_rows == 1 and missing.n_valid == 0
    for values in ([], [None] * 8, [np.inf] * 8):
        result = check(values)
        assert result.status == FlatlineStatus.INSUFFICIENT_SAMPLES
        assert result.longest_valid_run == 0
        assert not result.ongoing


def test_numeric_strings_nullable_values_and_nondefault_index():
    data = pd.DataFrame({"x": ["0", "0", "0", "0", pd.NA]}, index=[9, 3, 2, 7, 1])
    result = detect_flatlines(data, {"x": FlatlineConfig(0, 4)})[0]
    assert result.longest_flatline_run == 4
    assert result.n_valid == 4
    assert not result.ongoing
    result = check(pd.Series([2.0] * 4 + [pd.NA], dtype="Float64"))
    assert result.n_valid == 4


@pytest.mark.parametrize("values", [
    pd.date_range("2026-01-01", periods=5),
    pd.to_timedelta([1] * 5, unit="s"),
    [1 + 2j] * 5,
])
def test_non_real_amplitudes_are_not_valid_sensor_readings(values):
    result = check(values)
    assert result.n_valid == 0
    assert result.status == FlatlineStatus.INSUFFICIENT_SAMPLES


@pytest.mark.parametrize("tolerance,minimum", [
    (-1, 4), (np.nan, 4), (np.inf, 4), (True, 4),
    (0, 1), (0, 0), (0, 2.5), (0, True),
])
def test_invalid_configuration(tolerance, minimum):
    with pytest.raises(ValueError):
        FlatlineConfig(tolerance, minimum)


def test_duplicate_sensor_columns_rejected():
    with pytest.raises(ValueError, match="unique"):
        detect_flatlines(pd.DataFrame([[1, 2]], columns=["x", "x"]),
                         {"x": FlatlineConfig(0, 2)})


def test_linear_scan_matches_exhaustive_window_reference():
    # Exhaustive short sequences catch queue boundary errors, including gaps.
    for values in itertools.product([0.0, 1.0, 2.0, np.nan], repeat=4):
        for tolerance in (0.0, 1.0):
            longest = trailing = 0
            for start in range(len(values)):
                for end in range(start + 1, len(values) + 1):
                    window = np.asarray(values[start:end])
                    if np.isfinite(window).all() and np.ptp(window) <= tolerance:
                        longest = max(longest, end - start)
                        if end == len(values):
                            trailing = max(trailing, end - start)
            result = check(values, tolerance=tolerance, minimum=2)
            assert result.longest_flatline_run == longest
            assert result.trailing_flatline_run == trailing
            assert result.ongoing == (trailing >= 2)
