import numpy as np
import pandas as pd
import pytest

from pdm.data.leakage import detect_feature_leakage, format_leakage_report


def _findings(report, check=None, column=None):
    items = report["findings"]
    if check is not None:
        items = [item for item in items if item["check"] == check]
    if column is not None:
        items = [item for item in items if item["column"] == column]
    return items


def test_flags_monotonic_row_order_proxy_but_not_random_sensor():
    rng = np.random.default_rng(7)
    n = 500
    df = pd.DataFrame({
        "original_index": np.arange(n),
        "sensor": rng.normal(size=n),
        "label": rng.integers(0, 2, size=n),
    })

    report = detect_feature_leakage(df, target_columns="label")

    row_findings = _findings(report, check="row_order_correlation")
    assert any(item["column"] == "original_index" for item in row_findings)
    assert not any(item["column"] == "sensor" for item in row_findings)
    assert "original_index" in report["flagged_columns"]


def test_flags_inverse_monotonic_proxy_and_preserves_order_when_sampling():
    n = 10_000
    df = pd.DataFrame({
        "reverse_index": np.arange(n, 0, -1),
        "label": np.tile([0, 1], n // 2),
    })

    report = detect_feature_leakage(df, target_columns="label", sample_size=101)

    finding = _findings(report, check="row_order_correlation", column="reverse_index")[0]
    assert finding["correlation"] == pytest.approx(-1.0)
    assert report["sample_rows"] == 101
    assert report["total_rows"] == n


def test_flags_exact_target_copy_for_non_numeric_values():
    target = pd.Series(["ok", "fail"] * 50)
    df = pd.DataFrame({
        "status_copy": target.copy(),
        "sensor_name": [f"s{i % 5}" for i in range(len(target))],
        "target": target,
    })

    report = detect_feature_leakage(df, target_columns="target")

    findings = _findings(report, check="exact_target_match", column="status_copy")
    assert len(findings) == 1
    assert findings[0]["target"] == "target"
    assert findings[0]["score"] == 1.0


def test_flags_near_perfect_numeric_target_proxy():
    rng = np.random.default_rng(11)
    target = rng.integers(0, 2, size=500)
    df = pd.DataFrame({
        "future_failure_score": target * 100.0 + rng.normal(0, 0.01, size=len(target)),
        "sensor": rng.normal(size=len(target)),
        "label": target,
    })

    report = detect_feature_leakage(df, target_columns="label")

    findings = _findings(report, check="target_correlation", column="future_failure_score")
    assert len(findings) == 1
    assert findings[0]["target"] == "label"
    assert findings[0]["score"] >= 0.995
    assert not _findings(report, check="target_correlation", column="sensor")


def test_supports_multiple_targets_and_ignored_keys():
    n = 200
    df = pd.DataFrame({
        "device_id": np.arange(n),
        "observation_time": pd.date_range("2026-01-01", periods=n, freq="h"),
        "proxy_b": np.tile([0, 1], n // 2),
        "label_a": np.tile([1, 0], n // 2),
        "label_b": np.tile([0, 1], n // 2),
    })

    report = detect_feature_leakage(
        df,
        target_columns=["label_a", "label_b"],
        ignore_columns=["device_id", "observation_time"],
    )

    assert "device_id" not in report["flagged_columns"]
    assert "observation_time" not in report["flagged_columns"]
    exact = _findings(report, check="exact_target_match", column="proxy_b")
    assert len(exact) == 1
    assert exact[0]["target"] == "label_b"


def test_constant_and_all_null_columns_are_ignored_and_input_is_unchanged():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        "constant": 3.0,
        "all_null": np.nan,
        "sensor": rng.normal(size=100),
        "label": rng.integers(0, 2, size=100),
    })
    before = df.copy(deep=True)

    report = detect_feature_leakage(df, target_columns="label")

    assert "constant" not in report["flagged_columns"]
    assert "all_null" not in report["flagged_columns"]
    pd.testing.assert_frame_equal(df, before)


def test_numeric_strings_can_be_detected_as_row_order_proxy():
    n = 100
    df = pd.DataFrame({
        "sequence_id": [str(i) for i in range(n)],
        "label": np.tile([0, 1], n // 2),
    })

    report = detect_feature_leakage(df, target_columns="label")

    assert _findings(report, check="row_order_correlation", column="sequence_id")


def test_validation_errors_are_clear():
    df = pd.DataFrame({"x": range(20), "label": [0, 1] * 10})

    with pytest.raises(KeyError, match="Target columns not found"):
        detect_feature_leakage(df, target_columns="missing")
    with pytest.raises(ValueError, match="row_order_threshold"):
        detect_feature_leakage(df, target_columns="label", row_order_threshold=0)
    with pytest.raises(ValueError, match="target_correlation_threshold"):
        detect_feature_leakage(df, target_columns="label", target_correlation_threshold=1.1)
    with pytest.raises(ValueError, match="sample_size"):
        detect_feature_leakage(df, target_columns="label", sample_size=0)


def test_format_leakage_report_is_human_readable():
    clean = {
        "findings": [],
    }
    assert format_leakage_report(clean) == "Leakage check: no obvious leakage candidates found."

    report = detect_feature_leakage(
        pd.DataFrame({"original_index": range(50), "label": [0, 1] * 25}),
        target_columns="label",
    )
    text = format_leakage_report(report)
    assert "review before training" in text
    assert "original_index" in text
