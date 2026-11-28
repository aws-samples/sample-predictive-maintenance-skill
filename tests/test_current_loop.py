"""Tests for current-loop signal bands and sustained alarm diagnostics."""

import itertools

import numpy as np
import pandas as pd
import pytest

from pdm.monitoring.current_loop import (
    CurrentBand, CurrentLoopConfig, CurrentLoopStatus,
    classify_current, detect_current_loop,
)


CONFIG = CurrentLoopConfig.ne43_example(3)


def check(values, config=CONFIG):
    return detect_current_loop(pd.DataFrame({"loop_ma": values}),
                               {"loop_ma": config})[0]


@pytest.mark.parametrize("value,band", [
    (0, CurrentBand.LOW_ALARM),
    (3.6, CurrentBand.LOW_ALARM),
    (3.60001, CurrentBand.LOW_TRANSITION),
    (3.79999, CurrentBand.LOW_TRANSITION),
    (3.8, CurrentBand.UNDERRANGE),
    (3.99999, CurrentBand.UNDERRANGE),
    (4.0, CurrentBand.NOMINAL),
    (12.0, CurrentBand.NOMINAL),
    (20.0, CurrentBand.NOMINAL),
    (20.00001, CurrentBand.OVERRANGE),
    (20.5, CurrentBand.OVERRANGE),
    (20.50001, CurrentBand.HIGH_TRANSITION),
    (20.99999, CurrentBand.HIGH_TRANSITION),
    (21.0, CurrentBand.HIGH_ALARM),
    (23.0, CurrentBand.HIGH_ALARM),
    ("12.0", CurrentBand.NOMINAL),
])
def test_classification_boundaries(value, band):
    assert classify_current(value, CONFIG) == band


@pytest.mark.parametrize("value", [
    None, pd.NA, np.nan, np.inf, -np.inf, -0.01, "bad", True,
    np.bool_(False), 1 + 2j, pd.Timestamp("2026-01-01"), pd.Timedelta(seconds=1),
])
def test_invalid_nonunipolar_or_nonnumeric_values(value):
    assert classify_current(value, CONFIG) == CurrentBand.INVALID


def test_process_underrange_and_overrange_are_not_alarm():
    result = check([3.8, 3.9, 3.8, 4, 20, 20.1, 20.5, 20.5])
    assert result.status == CurrentLoopStatus.NO_SUSTAINED_ALARM
    assert result.n_measurement == 8
    assert result.bands[CurrentBand.UNDERRANGE].n_samples == 3
    assert result.bands[CurrentBand.OVERRANGE].n_samples == 3
    assert not result.ongoing


def test_transition_bands_are_neither_measurement_nor_alarm():
    result = check([3.7] * 4 + [20.7] * 4)
    assert result.status == CurrentLoopStatus.NO_SUSTAINED_ALARM
    assert result.n_valid == 8 and result.n_measurement == 0
    assert result.bands[CurrentBand.LOW_TRANSITION].n_samples == 4
    assert result.bands[CurrentBand.HIGH_TRANSITION].n_samples == 4
    assert not result.low_alarm_suspected and not result.high_alarm_suspected


def test_zero_current_is_low_alarm_not_missing():
    result = check([12, 0, 0, 0])
    assert result.status == CurrentLoopStatus.SUSPECTED_ALARM
    assert result.low_alarm_suspected and result.ongoing
    assert result.bands[CurrentBand.INVALID].n_samples == 0


def test_high_alarm_and_recovery():
    result = check([12, 21, 22, 23, 12])
    assert result.high_alarm_suspected
    assert result.status == CurrentLoopStatus.SUSPECTED_ALARM
    assert not result.ongoing
    assert result.bands[CurrentBand.HIGH_ALARM].longest_run == 3
    assert result.bands[CurrentBand.HIGH_ALARM].trailing_run == 0


@pytest.mark.parametrize("gap", [None, np.inf, "bad", -1, 3.7, 12, 20.7, 22])
def test_same_direction_alarm_run_broken_by_other_band(gap):
    result = check([0, 0, gap, 0, 0])
    assert not result.low_alarm_suspected
    assert result.bands[CurrentBand.LOW_ALARM].longest_run == 2
    assert not result.ongoing


def test_both_directions_and_final_invalid_sample():
    result = check([0, 1, 2, 12, 21, 22, 23, None])
    assert result.low_alarm_suspected and result.high_alarm_suspected
    assert not result.ongoing
    assert result.bands[CurrentBand.INVALID].trailing_run == 1


def test_alternating_alarm_directions_are_not_sustained_same_direction():
    result = check([0, 22] * 5)
    assert result.status == CurrentLoopStatus.NO_SUSTAINED_ALARM
    assert result.n_measurement == 0
    assert result.bands[CurrentBand.LOW_ALARM].n_samples == 5
    assert result.bands[CurrentBand.HIGH_ALARM].n_samples == 5


def test_missing_empty_and_unusable_channels():
    missing = detect_current_loop(pd.DataFrame({"other": [0, 1]}),
                                  {"loop_ma": CONFIG})[0]
    assert missing.status == CurrentLoopStatus.MISSING_COLUMN
    assert missing.n_rows == 2 and missing.n_valid == 0
    assert sum(b.n_samples for b in missing.bands.values()) == 0
    for values in ([], [None] * 4, [12, None, 12, None, 12]):
        result = check(values)
        assert result.status == CurrentLoopStatus.INSUFFICIENT_SAMPLES
        assert not result.ongoing


def test_custom_profile_and_channel_independence():
    custom = CurrentLoopConfig(3.5, 3.9, 4, 20, 20.4, 22, 3)
    data = pd.DataFrame({"a": [21.5] * 4, "b": [21.5] * 4},
                        index=[4, 1, 7, 2])
    a, b = detect_current_loop(data, {"a": CONFIG, "b": custom})
    assert a.high_alarm_suspected
    assert b.bands[CurrentBand.HIGH_TRANSITION].n_samples == 4
    assert not b.high_alarm_suspected
    assert detect_current_loop(data, {}) == []


def test_nullable_numeric_inputs():
    result = check(pd.Series([0, 0, 0, pd.NA], dtype="Float64"))
    assert result.low_alarm_suspected and not result.ongoing
    assert result.n_valid == 3


@pytest.mark.parametrize("args", [
    (-1, 3.8, 4, 20, 20.5, 21, 3),
    (3.8, 3.6, 4, 20, 20.5, 21, 3),
    (3.6, 4, 4, 20, 20.5, 21, 3),
    (3.6, 3.8, 4, 20, 21, 21, 3),
    (np.nan, 3.8, 4, 20, 20.5, 21, 3),
    (3.6, 3.8, 4, 20, 20.5, np.inf, 3),
    (False, 3.8, 4, 20, 20.5, 21, 3),
    (3.6, 3.8, 4, 20, 20.5, 21, 1),
    (3.6, 3.8, 4, 20, 20.5, 21, 2.5),
    (3.6, 3.8, 4, 20, 20.5, 21, True),
])
def test_invalid_config(args):
    with pytest.raises(ValueError):
        CurrentLoopConfig(*args)


def test_invalid_columns_and_config_types():
    with pytest.raises(ValueError, match="unique"):
        detect_current_loop(pd.DataFrame([[0, 1]], columns=["a", "a"]), {"a": CONFIG})
    with pytest.raises(TypeError, match="CurrentLoopConfig"):
        detect_current_loop(pd.DataFrame({"a": [12]}), {"a": {}})


def test_scan_matches_independent_groupby_reference():
    values_by_band = [
        (None, CurrentBand.INVALID), (0, CurrentBand.LOW_ALARM),
        (3.7, CurrentBand.LOW_TRANSITION), (3.9, CurrentBand.UNDERRANGE),
        (12, CurrentBand.NOMINAL), (20.3, CurrentBand.OVERRANGE),
        (20.7, CurrentBand.HIGH_TRANSITION), (22, CurrentBand.HIGH_ALARM),
    ]
    for sequence in itertools.product(values_by_band, repeat=3):
        values, expected = zip(*sequence)
        result = check(values)
        for band, activity in result.bands.items():
            groups = [list(group) for key, group in itertools.groupby(expected)
                      if key == band]
            assert activity.n_samples == sum(map(len, groups))
            assert activity.longest_run == max(map(len, groups), default=0)
            assert activity.trailing_run == (len(groups[-1]) if expected[-1] == band else 0)
        assert sum(b.n_samples for b in result.bands.values()) == len(values)
        assert result.n_valid == len(values) - expected.count(CurrentBand.INVALID)
