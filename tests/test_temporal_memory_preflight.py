"""Regression tests for the temporal sliding-window memory pre-flight."""

import warnings

import numpy as np
import pandas as pd
import pytest

import pdm.anomaly_detection.temporal as temporal
from pdm.anomaly_detection.temporal import (
    TemporalAnomalyDetector,
    estimate_window_memory_bytes,
)


def test_estimate_counts_complete_float32_windows():
    # 100 timesteps, 10-step window, 200 sensors: 91 complete windows.
    assert estimate_window_memory_bytes(100, 200, 10) == 91 * 10 * 200 * 4
    assert estimate_window_memory_bytes(3, 10, 5) == 0


@pytest.mark.parametrize(
    "args", [(-1, 2, 2, 4), (10, -1, 2, 4), (10, 2, 0, 4), (10, 2, 2, 0)]
)
def test_estimate_rejects_invalid_dimensions(args):
    with pytest.raises(ValueError):
        estimate_window_memory_bytes(*args)


def test_small_windows_do_not_warn(monkeypatch):
    monkeypatch.setattr(temporal, "WINDOW_MEMORY_WARNING_BYTES", 1024)
    model = TemporalAnomalyDetector()
    data = np.zeros((6, 4), dtype=np.float32)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        output = model._create_windows(data, 3)
    assert output.shape == (4, 12)
    assert not [w for w in captured if issubclass(w.category, RuntimeWarning)]


def test_large_windows_warn_and_preserve_output(monkeypatch):
    monkeypatch.setattr(temporal, "WINDOW_MEMORY_WARNING_BYTES", 100)
    model = TemporalAnomalyDetector()
    data = np.arange(24, dtype=np.float32).reshape(6, 4)
    with pytest.warns(RuntimeWarning, match="Reduce window_size"):
        output = model._create_windows(data, 3)
    assert output.shape == (4, 12)
    np.testing.assert_array_equal(output[0], data[:3].T.reshape(-1))


def test_warning_precedes_window_materialization(monkeypatch):
    monkeypatch.setattr(temporal, "WINDOW_MEMORY_WARNING_BYTES", 1)

    def prevent_allocation(*args, **kwargs):
        raise RuntimeError("allocation was attempted")

    monkeypatch.setattr(np.lib.stride_tricks, "sliding_window_view", prevent_allocation)
    model = TemporalAnomalyDetector()
    with pytest.warns(RuntimeWarning, match="sliding-window matrix"):
        with pytest.raises(RuntimeError, match="allocation was attempted"):
            model._create_windows(np.zeros((6, 4), dtype=np.float32), 3)


def test_estimate_uses_actual_dtype_when_guarding(monkeypatch):
    # Same shape: float32 needs 240 bytes, float64 needs 480 bytes.
    monkeypatch.setattr(temporal, "WINDOW_MEMORY_WARNING_BYTES", 300)
    model = TemporalAnomalyDetector()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model._create_windows(np.zeros((8, 3), dtype=np.float32), 4)
    assert not captured
    with pytest.warns(RuntimeWarning, match="sliding-window matrix"):
        model._create_windows(np.zeros((8, 3), dtype=np.float64), 4)


def test_predict_and_explain_share_the_preflight(monkeypatch):
    rng = np.random.default_rng(42)
    train_df = pd.DataFrame(rng.normal(size=(30, 3)), columns=list("abc"))
    test_df = pd.DataFrame(rng.normal(size=(20, 3)), columns=list("abc"))
    model = TemporalAnomalyDetector(window_size=3, smooth_window=3)
    model.train(train_df, test_df)
    monkeypatch.setattr(temporal, "WINDOW_MEMORY_WARNING_BYTES", 1)
    with pytest.warns(RuntimeWarning, match="sliding-window matrix"):
        assert len(model.predict(test_df).predictions) == len(test_df)
    with pytest.warns(RuntimeWarning, match="sliding-window matrix"):
        assert len(model.explain(test_df, top_k=2)) == len(test_df)
