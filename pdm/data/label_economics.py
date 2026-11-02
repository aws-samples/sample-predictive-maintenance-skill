"""Label-economics diagnostics for classification formulation guidance."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

DEFAULT_SPARSE_POSITIVE_RATE = 0.01
DEFAULT_FEW_EVENT_THRESHOLD = 20


def _as_list(columns: str | Iterable[str]) -> list[str]:
    if isinstance(columns, str):
        return [columns]
    return list(columns)


def _binary_indicator(series: pd.Series, label: str) -> pd.Series:
    """Return a nullable integer 0/1 view of a binary label column."""
    non_null = series.dropna()
    converted = pd.to_numeric(non_null, errors="coerce")
    if converted.isna().any() or not converted.isin([0, 1]).all():
        values = sorted({str(v) for v in non_null.unique()})[:10]
        raise ValueError(f"Label {label!r} must be binary 0/1; observed values: {values}")

    result = pd.Series(pd.NA, index=series.index, dtype="Int64", name=series.name)
    result.loc[non_null.index] = converted.astype("int64")
    return result


def _count_positive_events(
    df: pd.DataFrame,
    label: str,
    indicator: pd.Series,
    *,
    time_col: str | None,
    entity_col: str | None,
) -> int | None:
    """Count contiguous positive runs after sorting by entity/time.

    A distinct event is a 0 -> 1 transition (or a positive first observation)
    within each entity. Event counts are only reported when a time column is
    supplied; without temporal ordering, row adjacency has no reliable event
    meaning.
    """
    if time_col is None:
        return None

    columns = [time_col]
    if entity_col is not None:
        columns.append(entity_col)
    work = df[columns].copy()
    work["__label"] = indicator
    work = work.dropna(subset=[time_col, "__label"])
    if work.empty:
        return 0

    sort_cols = [time_col] if entity_col is None else [entity_col, time_col]
    work = work.sort_values(sort_cols, kind="stable")
    positive = work["__label"].astype(bool)

    if entity_col is None:
        previous = positive.shift(fill_value=False)
    else:
        previous = positive.groupby(work[entity_col], sort=False).shift(fill_value=False)

    return int((positive & ~previous).sum())


def analyze_classification_label_economics(
    df: pd.DataFrame,
    label_columns: str | Iterable[str],
    *,
    entity_col: str | None = None,
    time_col: str | None = None,
    sparse_positive_rate: float = DEFAULT_SPARSE_POSITIVE_RATE,
    few_event_threshold: int = DEFAULT_FEW_EVENT_THRESHOLD,
) -> dict[str, Any]:
    """Summarize classification-label scarcity and recommend a modeling role.

    This helper is intended for Phase 2 formulation guidance. It measures each
    binary label's positive prevalence and, when temporal ordering is available,
    the number of distinct positive events (contiguous positive runs). A long
    incident can therefore contribute many positive rows but still only one
    independent failure event.

    The recommendation is advisory only:

    * ``anomaly_primary`` when every label is very sparse or few-event;
    * ``hybrid`` when only some labels meet those conditions; and
    * ``classification_primary_candidate`` when none do.

    The caller must still consider censoring, run-to-failure structure, label
    completeness, and operational requirements before the user confirms a final
    formulation.
    """
    if not 0.0 < sparse_positive_rate < 1.0:
        raise ValueError("sparse_positive_rate must be in (0, 1)")
    if few_event_threshold < 1:
        raise ValueError("few_event_threshold must be at least 1")

    labels = _as_list(label_columns)
    if not labels:
        raise ValueError("label_columns must contain at least one label")

    required = labels + [c for c in (entity_col, time_col) if c is not None]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")

    per_label: dict[str, dict[str, Any]] = {}
    flagged_labels: list[str] = []

    for label in labels:
        indicator = _binary_indicator(df[label], label)
        non_null = indicator.notna()
        total = int(non_null.sum())
        positives = int((indicator == 1).sum())
        rate = positives / total if total else 0.0
        events = _count_positive_events(
            df,
            label,
            indicator,
            time_col=time_col,
            entity_col=entity_col,
        )

        positive_entities = None
        if entity_col is not None:
            positive_mask = indicator == 1
            positive_entities = int(df.loc[positive_mask.fillna(False), entity_col].nunique(dropna=True))

        sparse = rate < sparse_positive_rate
        few_events = events is not None and events < few_event_threshold
        label_flagged = sparse or few_events
        if label_flagged:
            flagged_labels.append(label)

        reasons: list[str] = []
        if sparse:
            reasons.append(
                f"positive rate {rate:.3%} is below the {sparse_positive_rate:.1%} sparse-label threshold"
            )
        if few_events:
            reasons.append(
                f"{events} distinct positive events is below the {few_event_threshold}-event screening threshold"
            )

        per_label[label] = {
            "rows_with_label": total,
            "positives": positives,
            "positive_rate": round(rate, 8),
            "positive_entities": positive_entities,
            "distinct_positive_events": events,
            "sparse_positive_rate": sparse,
            "few_distinct_events": few_events,
            "reasons": reasons,
        }

    if len(flagged_labels) == len(labels):
        recommendation = "anomaly_primary"
        guidance = (
            "All selected classification labels are sparse and/or supported by few distinct events. "
            "Recommend anomaly detection as the primary formulation and supervised classification "
            "as a secondary check for the known failure modes."
        )
    elif flagged_labels:
        recommendation = "hybrid"
        guidance = (
            "Only some classification labels are sparse and/or few-event. Recommend a hybrid strategy: "
            "keep supervised classification for labels with adequate support and add anomaly detection "
            "for sparse labels and novel failure modes."
        )
    else:
        recommendation = "classification_primary_candidate"
        guidance = (
            "Label economics do not indicate severe scarcity. Supervised classification can remain the "
            "primary candidate, with anomaly detection considered separately for novel or unlabeled failures."
        )

    return {
        "sparse_positive_rate_threshold": float(sparse_positive_rate),
        "few_event_threshold": int(few_event_threshold),
        "event_count_available": time_col is not None,
        "entity_col": entity_col,
        "time_col": time_col,
        "labels": per_label,
        "flagged_labels": flagged_labels,
        "recommendation": recommendation,
        "guidance": guidance,
    }


def format_label_economics_report(report: dict[str, Any]) -> str:
    """Format label-economics diagnostics for exploration logs or reports."""
    lines = ["Classification label economics:"]
    for label, stats in report["labels"].items():
        line = f"- {label}: {stats['positives']}/{stats['rows_with_label']} positive ({stats['positive_rate']:.3%})"
        if stats["distinct_positive_events"] is not None:
            line += f", {stats['distinct_positive_events']} distinct event(s)"
        if stats["positive_entities"] is not None:
            line += f", {stats['positive_entities']} positive entity/entities"
        if stats["reasons"]:
            line += " — " + "; ".join(stats["reasons"])
        lines.append(line)
    lines.append(f"Recommendation: {report['guidance']}")
    return "\n".join(lines)
