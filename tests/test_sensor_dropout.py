"""Tests for bounded intermittent and sustained sensor dropout diagnostics."""

import itertools

import numpy as np
import pandas as pd
import pytest

from pdm.monitoring.dropout import (
    DropoutConfig, DropoutStatus, detect_dropouts,
)


CONFIG = DropoutConfig(min_recovered_episodes=2, sustained_run_samples=3)


def check(values, config=CONFIG):
    return detect_dropouts(pd.DataFrame({"sensor": values}), {"sensor": config})[0]


def test_repeated_short_dropouts_and_recovery():
    result = check([12, None, 12, None, "bad", 12])
    assert result.status == DropoutStatus.SUSPECTED_INTERMITTENT
    assert result.n_episodes == result.n_recovered_short_episodes == 2
    assert result.n_sustained_episodes == 0
    assert (result.n_missing, result.n_invalid, result.n_valid) == (2, 1, 3)
    assert result.dropout_fraction == 0.5
    assert result.longest_dropout_run == 2
    assert result.intermittent_suspected
    assert not result.ongoing_dropout


def test_single_sustained_episode_not_intermittent():
    result = check([12, None, np.inf, None, 12])
    assert result.status == DropoutStatus.SUSPECTED_SUSTAINED
    assert result.n_episodes == result.n_sustained_episodes == 1
    assert result.n_recovered_short_episodes == 0
    assert result.sustained_suspected and not result.ongoing_sustained


def test_mixed_patterns_and_ongoing_sustained_dropout():
    result = check([1, None, 1, None, 1, None, None, None])
    assert result.status == DropoutStatus.SUSPECTED_BOTH
    assert result.intermittent_suspected and result.sustained_suspected
    assert result.ongoing_sustained
    assert result.trailing_dropout_run == 3


def test_window_boundaries_do_not_count_as_bounded_short_episodes():
    result = check([None, 1, None, 1, None])
    assert result.n_episodes == 3
    assert result.n_recovered_short_episodes == 1
    assert result.leading_dropout_run == result.trailing_dropout_run == 1
    assert not result.intermittent_suspected
    assert result.ongoing_dropout and not result.ongoing_sustained


def test_leading_sustained_run_is_observable_even_if_left_censored():
    result = check([None, None, None, 1, 2])
    assert result.leading_dropout_run == 3
    assert result.sustained_suspected
    assert not result.ongoing_dropout


def test_all_unusable_channel_does_not_double_count_boundaries():
    result = check([None, "bad", np.inf, pd.NA])
    assert result.status == DropoutStatus.NO_VALID_SAMPLES
    assert result.n_episodes == result.n_sustained_episodes == 1
    assert result.leading_dropout_run == result.trailing_dropout_run == 4
    assert result.n_recovered_short_episodes == 0
    assert result.dropout_fraction == 1
    assert result.ongoing_sustained


def test_all_unusable_short_window_is_not_sustained():
    result = check([None, None])
    assert result.status == DropoutStatus.NO_VALID_SAMPLES
    assert result.ongoing_dropout and not result.sustained_suspected


def test_zero_negative_and_numeric_strings_are_not_dropouts():
    result = check([0, -2, "0", "-3.5", 2])
    assert result.status == DropoutStatus.NO_QUALIFYING_DROPOUT
    assert result.n_valid == 5
    assert result.dropout_fraction == 0
    assert result.n_episodes == 0


def test_invalid_analog_types_are_not_valid():
    result = check(pd.Series(
        [1, True, np.bool_(False), 1 + 2j, pd.Timestamp("2026-01-01"),
         pd.Timedelta(seconds=1), "nan", "inf", 1], dtype=object))
    assert (result.n_valid, result.n_invalid, result.n_missing) == (2, 7, 0)
    assert result.n_sustained_episodes == 1


def test_nullable_numeric_and_nondefault_row_order():
    frame = pd.DataFrame({"x": pd.array([1, pd.NA, 1, pd.NA, 1], dtype="Float64")},
                         index=[9, 2, 7, 1, 0])
    result = detect_dropouts(frame, {"x": CONFIG})[0]
    assert result.intermittent_suspected
    assert result.n_missing == 2


def test_missing_column_and_empty_window_are_distinct():
    missing = detect_dropouts(pd.DataFrame({"other": [1, 2]}), {"sensor": CONFIG})[0]
    assert missing.status == DropoutStatus.MISSING_COLUMN
    assert missing.n_rows == 2
    assert missing.dropout_fraction is None
    assert missing.n_episodes == 0
    assert not missing.ongoing_dropout
    empty = check([])
    assert empty.status == DropoutStatus.EMPTY_WINDOW
    assert empty.dropout_fraction is None


def test_insufficient_window():
    assert check([1, None]).status == DropoutStatus.INSUFFICIENT_SAMPLES


def test_custom_thresholds_and_independent_channels():
    data = pd.DataFrame({"a": [1, None, 1, None, 1], "b": [1, None, 1, None, 1]})
    a, b = detect_dropouts(data, {"a": CONFIG, "b": DropoutConfig(3, 3)})
    assert a.intermittent_suspected and not b.intermittent_suspected
    assert detect_dropouts(data, {}) == []


@pytest.mark.parametrize("args", [(0, 3), (1, 3), (2, 1), (2, -1),
                                  (2.5, 3), (2, 3.5), (True, 3), (2, np.bool_(True))])
def test_invalid_configuration(args):
    with pytest.raises(ValueError):
        DropoutConfig(*args)


def test_duplicate_columns_and_wrong_configuration_rejected():
    with pytest.raises(ValueError, match="unique"):
        detect_dropouts(pd.DataFrame([[1, 2]], columns=["x", "x"]), {"x": CONFIG})
    with pytest.raises(TypeError, match="DropoutConfig"):
        detect_dropouts(pd.DataFrame({"x": [1]}), {"x": {}})


def test_scan_matches_independent_groupby_reference():
    for sequence in itertools.product([1.0, None, "bad"], repeat=5):
        result = check(sequence)
        flags = [value != 1.0 for value in sequence]
        groups = [(key, len(list(group))) for key, group in itertools.groupby(flags)]
        runs = [size for gap, size in groups if gap]
        bounded = [size for gap, size in groups[1:-1] if gap and size < 3]
        assert result.n_episodes == len(runs)
        assert result.longest_dropout_run == max(runs, default=0)
        assert result.n_recovered_short_episodes == len(bounded)
        assert result.n_sustained_episodes == sum(size >= 3 for size in runs)
        assert result.leading_dropout_run == (groups[0][1] if groups[0][0] else 0)
        assert result.trailing_dropout_run == (groups[-1][1] if groups[-1][0] else 0)
        assert result.n_valid + result.n_missing + result.n_invalid == len(sequence)
