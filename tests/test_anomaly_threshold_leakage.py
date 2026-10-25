"""Regression tests for held-out-label leakage in anomaly thresholds."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pdm.anomaly_detection.spectral_residual import SpectralResidualDetector
from pdm.anomaly_detection.temporal import TemporalAnomalyDetector
from pdm.benchmarks.benchmark import run_smap


@pytest.fixture
def threshold_data():
    """Normal train data plus identical held-out features with two labelings."""
    rng = np.random.default_rng(42)
    columns = [f"sensor_{i}" for i in range(4)]
    train_df = pd.DataFrame(rng.normal(size=(120, 4)), columns=columns)
    test_features = pd.DataFrame(rng.normal(size=(80, 4)), columns=columns)

    labels_a = np.zeros(80, dtype=int)
    labels_a[10:20] = 1
    labels_b = np.zeros(80, dtype=int)
    labels_b[45:70] = 1

    test_a = test_features.copy()
    test_a["label"] = labels_a
    test_b = test_features.copy()
    test_b["label"] = labels_b
    return train_df, test_a, test_b


@pytest.mark.parametrize(
    ("detector_cls", "kwargs"),
    [
        (TemporalAnomalyDetector, {"window_size": 4, "smooth_window": 3}),
        (SpectralResidualDetector, {"sr_window": 5, "aggregation_percentile": 95}),
    ],
)
def test_test_labels_do_not_influence_threshold(
    detector_cls, kwargs, threshold_data, monkeypatch
):
    """Changing held-out labels must not change persisted/deployable threshold."""
    train_df, test_a, test_b = threshold_data

    def fail_if_called(*args, **kwargs):
        raise AssertionError("train() must not optimize thresholds on held-out labels")

    monkeypatch.setattr(detector_cls, "_optimize_threshold", fail_if_called)

    first = detector_cls(contamination=0.10, **kwargs)
    second = detector_cls(contamination=0.10, **kwargs)

    first_result = first.train(train_df, test_a)
    second_result = second.train(train_df, test_b)

    assert first.threshold == pytest.approx(second.threshold)
    assert first_result.metrics["threshold"] == pytest.approx(second_result.metrics["threshold"])


def test_smap_benchmark_does_not_use_test_prevalence_for_contamination(
    tmp_path, monkeypatch
):
    """SMAP labels may score the benchmark, but must not configure its threshold."""
    train_df = pd.DataFrame({"sensor_0": np.linspace(0.0, 1.0, 12)})
    test_df = pd.DataFrame({
        "sensor_0": np.linspace(1.0, 2.0, 10),
        # Deliberately 70% positive: old code would pass min(0.70, 0.15) == 0.15.
        "label": [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    })
    train_df.to_csv(tmp_path / "raw_train.csv", index=False)
    test_df.to_csv(tmp_path / "raw_test.csv", index=False)

    observed_contamination = []

    class FakeSpectralResidualDetector:
        def __init__(
            self,
            sr_window=5,
            aggregation_percentile=95,
            contamination=0.10,
            scoring="percentile",
        ):
            observed_contamination.append(contamination)
            self.feature_names = ["sensor_0"]

        def train(self, train_df, test_df, **kwargs):
            return SimpleNamespace(metrics={})

        def predict(self, features):
            predictions = pd.DataFrame(
                {
                    "anomaly_score": np.linspace(0.0, 1.0, len(features)),
                    "is_anomaly": np.zeros(len(features), dtype=int),
                },
                index=features.index,
            )
            return SimpleNamespace(predictions=predictions)

    import pdm.anomaly_detection.spectral_residual as spectral_module

    monkeypatch.setattr(
        spectral_module,
        "SpectralResidualDetector",
        FakeSpectralResidualDetector,
    )

    metrics = run_smap(tmp_path)

    assert observed_contamination == [0.10]
    assert {"f1", "precision", "recall", "roc_auc", "average_precision"} <= metrics.keys()
