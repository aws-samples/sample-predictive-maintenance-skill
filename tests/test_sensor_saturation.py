"""Regression tests for configured ADC rail occupancy diagnostics."""

import itertools

import numpy as np
import pandas as pd
import pytest

from pdm.monitoring.saturation import (
    SaturationConfig, SaturationStatus, detect_saturation,
)


def check(values, *, lower=0, upper=4095, minimum=3, tolerance=0):
    config = SaturationConfig(lower, upper, minimum, tolerance)
    return detect_saturation(pd.DataFrame({"adc": values}), {"adc": config})[0]


def test_normal_adc_signal_and_isolated_boundary_hits():
    result = check([0, 200, 2000, 3000, 4095, 1000])
    assert result.status == SaturationStatus.NO_SATURATION_DETECTED
    assert result.lower.n_samples == result.upper.n_samples == 1
    assert result.rail_fraction == pytest.approx(2 / 6)
    assert result.n_in_range == 6
    assert not result.ongoing


def test_upper_saturation_and_recovery():
    result = check([100, 4095, 4095, 4095, 3000])
    assert result.status == SaturationStatus.SUSPECTED_SATURATION
    assert result.upper.suspected
    assert result.upper.longest_run == 3
    assert result.upper.trailing_run == 0
    assert not result.lower.suspected
    assert not result.ongoing


def test_lower_saturation_ongoing_at_zero():
    result = check([1000, 0, 0, 0, 0])
    assert result.status == SaturationStatus.SUSPECTED_SATURATION
    assert result.lower.longest_run == result.lower.trailing_run == 4
    assert result.lower.ongoing and result.ongoing
    assert result.n_invalid == 0


def test_bipolar_voltage_both_rails_independently():
    result = check([-10, -10, -10, 0, 10, 10, 10],
                   lower=-10, upper=10)
    assert result.lower.suspected and result.upper.suspected
    assert not result.lower.ongoing and result.upper.ongoing
    assert result.rail_fraction == pytest.approx(6 / 7)


def test_near_rail_band_includes_boundary():
    result = check([4093, 4094, 4095], tolerance=2)
    assert result.upper.suspected
    assert result.upper.n_samples == 3
    assert not check([4093, 4094, 4095], tolerance=1).upper.suspected
    assert check([0, 1, 2], tolerance=2).lower.suspected


def test_alternating_rails_do_not_create_same_rail_run():
    result = check([0, 4095] * 4)
    assert result.rail_fraction == 1.0
    assert result.lower.longest_run == result.upper.longest_run == 1
    assert result.status == SaturationStatus.NO_SATURATION_DETECTED


@pytest.mark.parametrize("gap", [None, np.nan, np.inf, -np.inf, "bad", pd.NA])
def test_invalid_gap_breaks_sustained_run(gap):
    result = check([4095, 4095, gap, 4095, 4095])
    assert result.n_invalid == 1 and result.n_in_range == 4
    assert result.upper.longest_run == result.upper.trailing_run == 2
    assert result.status == SaturationStatus.INSUFFICIENT_SAMPLES
    assert not result.ongoing


def test_outside_range_separate_from_rail_hits_and_breaks_run():
    result = check([0, 0, -1, 0, 0, 4096, 4095, 4095])
    assert result.n_below_range == result.n_above_range == 1
    assert result.n_invalid == 0
    assert result.n_in_range == 6
    assert result.lower.n_samples == 4 and result.upper.n_samples == 2
    assert result.lower.longest_run == result.upper.longest_run == 2
    assert result.rail_fraction == 1
    assert result.status == SaturationStatus.INSUFFICIENT_SAMPLES


def test_last_invalid_or_outside_sample_ends_ongoing_alert():
    for gap in (None, 4096):
        result = check([4095, 4095, 4095, gap])
        assert result.upper.suspected
        assert not result.ongoing
        assert result.upper.trailing_run == 0


def test_outside_only_window_has_no_in_range_denominator():
    result = check([-1, 4096, -2, 5000])
    assert result.n_below_range == result.n_above_range == 2
    assert result.n_in_range == 0
    assert result.rail_fraction is None
    assert result.status == SaturationStatus.INSUFFICIENT_SAMPLES


@pytest.mark.parametrize("values", [[], [None] * 4, [np.inf] * 4])
def test_empty_or_unusable_windows(values):
    result = check(values)
    assert result.status == SaturationStatus.INSUFFICIENT_SAMPLES
    assert result.n_in_range == 0
    assert result.rail_fraction is None
    assert not result.ongoing


def test_missing_column_and_unconfigured_channels():
    data = pd.DataFrame({"actual": [4095] * 4})
    result = detect_saturation(data, {"absent": SaturationConfig(0, 4095, 3)})[0]
    assert result.status == SaturationStatus.MISSING_COLUMN
    assert result.n_rows == 4
    assert result.n_in_range == result.n_invalid == 0
    assert result.rail_fraction is None
    assert detect_saturation(data, {}) == []


def test_numeric_strings_nullable_samples_and_row_order():
    data = pd.DataFrame({"adc": ["4095", "4095", "4095", pd.NA]},
                        index=[8, 3, 1, 5])
    result = detect_saturation(data, {"adc": SaturationConfig(0, 4095, 3)})[0]
    assert result.upper.suspected
    assert not result.ongoing
    assert result.n_invalid == 1
    result = check(pd.Series([0, 0, 0, pd.NA], dtype="Int64"))
    assert result.lower.suspected and result.n_invalid == 1


@pytest.mark.parametrize("values", [
    pd.date_range("2026-01-01", periods=4),
    pd.to_timedelta([1] * 4, unit="s"),
    [1 + 2j] * 4,
    [True] * 4,
])
def test_non_analog_values_are_not_rail_hits(values):
    result = check(values)
    assert result.n_invalid == 4
    assert result.n_in_range == 0
    assert result.status == SaturationStatus.INSUFFICIENT_SAMPLES


def test_channel_configurations_and_order_are_independent():
    frame = pd.DataFrame({"raw_adc": [4095] * 4, "voltage": [4.0, 4.1, 4.2, 4.3]})
    results = detect_saturation(frame, {
        "voltage": SaturationConfig(-10, 10, 3),
        "raw_adc": SaturationConfig(0, 4095, 4),
    })
    assert [r.feature for r in results] == ["voltage", "raw_adc"]
    assert results[0].status == SaturationStatus.NO_SATURATION_DETECTED
    assert results[1].upper.ongoing


@pytest.mark.parametrize("lower,upper,minimum,tolerance", [
    (10, 0, 3, 0), (0, 0, 3, 0), (np.nan, 10, 3, 0),
    (0, np.inf, 3, 0), (0, 10, 3, -1), (0, 10, 3, 5),
    (0, 10, 3, 6), (0, 10, 3, np.nan), (0, 10, 3, np.inf),
    (0, 10, 1, 0), (0, 10, 2.5, 0), (0, 10, True, 0),
    (False, 10, 3, 0), (0, 10, 3, True), ("0", 10, 3, 0),
])
def test_invalid_configuration(lower, upper, minimum, tolerance):
    with pytest.raises(ValueError):
        SaturationConfig(lower, upper, minimum, tolerance)


def test_duplicate_columns_and_wrong_config_rejected():
    with pytest.raises(ValueError, match="unique"):
        detect_saturation(pd.DataFrame([[0, 1]], columns=["adc", "adc"]),
                          {"adc": SaturationConfig(0, 4095, 3)})
    with pytest.raises(TypeError, match="SaturationConfig"):
        detect_saturation(pd.DataFrame({"adc": [0]}), {"adc": {}})


def test_scan_matches_independent_run_length_reference():
    # Exhaustively exercise transitions between both rails, interior, gaps
    # and out-of-range readings for short ordered windows.
    for values in itertools.product([0, 1, 2, np.nan, 3], repeat=4):
        result = check(values, lower=0, upper=2, minimum=2)
        for rail, code in ((result.lower, 0), (result.upper, 2)):
            groups = [list(group) for is_rail, group in
                      itertools.groupby(values, key=lambda v: v == code) if is_rail]
            assert rail.n_samples == sum(map(len, groups))
            assert rail.longest_run == max(map(len, groups), default=0)
            assert rail.trailing_run == (len(groups[-1]) if values[-1] == code else 0)
        assert (result.n_in_range + result.n_invalid +
                result.n_below_range + result.n_above_range) == len(values)
