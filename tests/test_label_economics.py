import pandas as pd
import pytest

from pdm.data.label_economics import (
    analyze_classification_label_economics,
    format_label_economics_report,
)


def test_counts_distinct_positive_runs_per_entity_after_time_sort():
    df = pd.DataFrame({
        "device_id": ["a", "a", "a", "a", "b", "b", "b", "b"],
        "timestamp": [4, 1, 2, 3, 4, 1, 3, 2],
        "label": [0, 1, 1, 0, 1, 0, 0, 1],
    })

    report = analyze_classification_label_economics(
        df,
        "label",
        entity_col="device_id",
        time_col="timestamp",
        sparse_positive_rate=0.01,
        few_event_threshold=4,
    )

    stats = report["labels"]["label"]
    assert stats["positives"] == 4
    assert stats["positive_rate"] == pytest.approx(0.5)
    assert stats["positive_entities"] == 2
    assert stats["distinct_positive_events"] == 3
    assert stats["few_distinct_events"] is True
    assert report["recommendation"] == "anomaly_primary"


def test_long_positive_incident_counts_as_one_event_not_many_rows():
    df = pd.DataFrame({
        "timestamp": range(100),
        "label": [0] * 40 + [1] * 50 + [0] * 10,
    })

    report = analyze_classification_label_economics(
        df,
        "label",
        time_col="timestamp",
        sparse_positive_rate=0.01,
        few_event_threshold=20,
    )

    stats = report["labels"]["label"]
    assert stats["positives"] == 50
    assert stats["distinct_positive_events"] == 1
    assert stats["sparse_positive_rate"] is False
    assert stats["few_distinct_events"] is True
    assert report["recommendation"] == "anomaly_primary"


def test_sparse_rate_drives_recommendation_when_event_order_is_unavailable():
    df = pd.DataFrame({"label": [1] + [0] * 199})

    report = analyze_classification_label_economics(df, "label")

    stats = report["labels"]["label"]
    assert stats["positive_rate"] == pytest.approx(0.005)
    assert stats["distinct_positive_events"] is None
    assert report["event_count_available"] is False
    assert report["recommendation"] == "anomaly_primary"


def test_multilabel_mixed_support_recommends_hybrid_strategy():
    n = 200
    df = pd.DataFrame({
        "timestamp": range(n),
        "label_common": [0, 1] * (n // 2),
        "label_sparse": [1] + [0] * (n - 1),
    })

    report = analyze_classification_label_economics(
        df,
        ["label_common", "label_sparse"],
        time_col="timestamp",
        few_event_threshold=20,
    )

    assert report["flagged_labels"] == ["label_sparse"]
    assert report["recommendation"] == "hybrid"


def test_supported_labels_leave_classification_as_primary_candidate():
    n = 200
    df = pd.DataFrame({
        "timestamp": range(n),
        "label": [0, 1] * (n // 2),
    })

    report = analyze_classification_label_economics(
        df,
        "label",
        time_col="timestamp",
        few_event_threshold=20,
    )

    stats = report["labels"]["label"]
    assert stats["positive_rate"] == pytest.approx(0.5)
    assert stats["distinct_positive_events"] == 100
    assert stats["sparse_positive_rate"] is False
    assert stats["few_distinct_events"] is False
    assert report["recommendation"] == "classification_primary_candidate"


def test_missing_labels_are_excluded_from_rate_denominator():
    df = pd.DataFrame({
        "timestamp": [1, 2, 3, 4, 5],
        "label": [1, 0, None, None, 0],
    })

    report = analyze_classification_label_economics(df, "label", time_col="timestamp")

    stats = report["labels"]["label"]
    assert stats["rows_with_label"] == 3
    assert stats["positives"] == 1
    assert stats["positive_rate"] == pytest.approx(1 / 3)


def test_rejects_non_binary_labels():
    df = pd.DataFrame({"label": [0, 1, 2, 0]})

    with pytest.raises(ValueError, match="must be binary 0/1"):
        analyze_classification_label_economics(df, "label")


def test_input_frame_is_not_modified():
    df = pd.DataFrame({
        "device_id": ["a", "a", "b", "b"],
        "timestamp": [2, 1, 2, 1],
        "label": [1, 0, 1, 0],
    })
    original = df.copy(deep=True)

    analyze_classification_label_economics(
        df,
        "label",
        entity_col="device_id",
        time_col="timestamp",
    )

    pd.testing.assert_frame_equal(df, original)


def test_report_formatter_surfaces_economics_and_recommendation():
    df = pd.DataFrame({"timestamp": range(100), "label": [1] + [0] * 99})
    report = analyze_classification_label_economics(df, "label", time_col="timestamp")

    text = format_label_economics_report(report)

    assert "1/100 positive (1.000%)" in text
    assert "1 distinct event(s)" in text
    assert "Recommendation:" in text
