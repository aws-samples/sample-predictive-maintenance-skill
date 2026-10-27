"""Cheap, warning-only leakage diagnostics for candidate PdM modeling frames."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_ROW_ORDER_CORRELATION = 0.995
DEFAULT_TARGET_CORRELATION = 0.995
DEFAULT_SAMPLE_SIZE = 100_000
DEFAULT_MIN_PAIRS = 20


def _as_list(columns: str | Iterable[str] | None) -> list[str]:
    if columns is None:
        return []
    if isinstance(columns, str):
        return [columns]
    return list(columns)


def _sample_preserving_order(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """Evenly sample a frame without destroying its original row order."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if len(df) <= sample_size:
        return df
    positions = np.linspace(0, len(df) - 1, num=sample_size, dtype=np.int64)
    return df.iloc[np.unique(positions)]


def _numeric_view(series: pd.Series) -> pd.Series | None:
    """Return a float view suitable for correlations, or None when not numeric."""
    if pd.api.types.is_datetime64_any_dtype(series):
        converted = pd.to_datetime(series, errors="coerce")
        values = converted.astype("int64").astype("float64")
        values[converted.isna()] = np.nan
        return pd.Series(values, index=series.index, name=series.name)

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype("float64")

    # Numeric identifiers sometimes arrive as strings. Treat them as numeric only
    # when nearly every non-null value converts cleanly.
    non_null = series.notna().sum()
    if non_null == 0:
        return None
    converted = pd.to_numeric(series, errors="coerce")
    if converted.notna().sum() / non_null >= 0.95:
        return converted.astype("float64")
    return None


def _paired_numeric(
    left: pd.Series,
    right: pd.Series,
    min_pairs: int,
) -> tuple[pd.Series, pd.Series] | None:
    left_num = _numeric_view(left)
    right_num = _numeric_view(right)
    if left_num is None or right_num is None:
        return None
    mask = left_num.notna() & right_num.notna()
    if int(mask.sum()) < min_pairs:
        return None
    a = left_num.loc[mask]
    b = right_num.loc[mask]
    if a.nunique(dropna=True) < 2 or b.nunique(dropna=True) < 2:
        return None
    return a, b


def _rank_correlation(left: pd.Series, right: pd.Series) -> float | None:
    corr = left.rank(method="average").corr(right.rank(method="average"))
    if pd.isna(corr):
        return None
    return float(corr)


def _exact_match(left: pd.Series, right: pd.Series, min_pairs: int) -> bool:
    mask = left.notna() & right.notna()
    if int(mask.sum()) < min_pairs:
        return False
    a = left.loc[mask].reset_index(drop=True)
    b = right.loc[mask].reset_index(drop=True)
    try:
        if bool(a.eq(b).all()):
            return True
    except (TypeError, ValueError):
        pass
    # Catch the common case where the same identifier/label is represented with
    # different scalar dtypes (for example 1 vs "1").
    return bool(a.astype(str).eq(b.astype(str)).all())


def detect_feature_leakage(
    df: pd.DataFrame,
    target_columns: str | Iterable[str] | None = None,
    ignore_columns: str | Iterable[str] | None = None,
    *,
    row_order_threshold: float = DEFAULT_ROW_ORDER_CORRELATION,
    target_correlation_threshold: float = DEFAULT_TARGET_CORRELATION,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    min_pairs: int = DEFAULT_MIN_PAIRS,
) -> dict[str, Any]:
    """Flag obvious feature-leakage candidates without modifying the data.

    The detector intentionally focuses on cheap, high-signal checks that are safe
    to run during data exploration/raw aggregation:

    * near-perfect monotonic correlation with row order (index/time proxies);
    * exact copies of a target column; and
    * near-perfect numeric correlation with a target (Pearson or rank).

    Findings are warnings for human review, not automatic drop decisions. Known
    non-feature keys such as entity IDs or split timestamps can be supplied via
    ``ignore_columns``.

    Args:
        df: Candidate modeling frame in its current row order.
        target_columns: Target/label columns to exclude from feature checks and to
            use for target-leakage diagnostics.
        ignore_columns: Known non-feature keys that should not be checked.
        row_order_threshold: Absolute rank correlation needed to flag a feature as
            a row-order proxy. Defaults to 0.995.
        target_correlation_threshold: Absolute Pearson or rank correlation needed
            to flag a numeric feature/target pair. Defaults to 0.995.
        sample_size: Maximum rows to inspect. Large frames are evenly sampled so
            the original ordering is preserved.
        min_pairs: Minimum paired non-null rows required for a correlation check.

    Returns:
        JSON-safe dict containing ``findings`` and de-duplicated
        ``flagged_columns``. The input frame is never modified.
    """
    if not 0.0 < row_order_threshold <= 1.0:
        raise ValueError("row_order_threshold must be in (0, 1]")
    if not 0.0 < target_correlation_threshold <= 1.0:
        raise ValueError("target_correlation_threshold must be in (0, 1]")
    if min_pairs < 2:
        raise ValueError("min_pairs must be at least 2")

    targets = _as_list(target_columns)
    ignored = set(_as_list(ignore_columns))
    missing_targets = [c for c in targets if c not in df.columns]
    if missing_targets:
        raise KeyError(f"Target columns not found: {missing_targets}")

    sample = _sample_preserving_order(df, sample_size)
    feature_columns = [c for c in sample.columns if c not in set(targets) | ignored]
    findings: list[dict[str, Any]] = []

    # Row-order proxy check. Use the sampled frame's original order, not its index
    # labels, because index labels may have been reset or carry business meaning.
    row_order = pd.Series(np.arange(len(sample), dtype=float), index=sample.index)
    for column in feature_columns:
        paired = _paired_numeric(sample[column], row_order, min_pairs)
        if paired is None:
            continue
        values, order = paired
        corr = _rank_correlation(values, order)
        if corr is None or abs(corr) < row_order_threshold:
            continue
        findings.append({
            "column": column,
            "check": "row_order_correlation",
            "score": round(abs(corr), 6),
            "correlation": round(corr, 6),
            "target": None,
            "message": (
                f"{column!r} has near-perfect monotonic correlation with row order "
                f"(|rho|={abs(corr):.4f}); review it as a possible index/time proxy."
            ),
        })

    # Target leakage checks. Exact matches also work for categorical/string labels;
    # correlation checks are limited to numeric/datetime-compatible pairs.
    for target in targets:
        target_series = sample[target]
        for column in feature_columns:
            feature = sample[column]
            if _exact_match(feature, target_series, min_pairs):
                findings.append({
                    "column": column,
                    "check": "exact_target_match",
                    "score": 1.0,
                    "correlation": None,
                    "target": target,
                    "message": f"{column!r} exactly matches target {target!r} on paired non-null rows.",
                })
                continue

            paired = _paired_numeric(feature, target_series, min_pairs)
            if paired is None:
                continue
            values, target_values = paired
            pearson = values.corr(target_values)
            rank_corr = _rank_correlation(values, target_values)
            candidates = [float(c) for c in (pearson, rank_corr) if c is not None and not pd.isna(c)]
            if not candidates:
                continue
            best = max(candidates, key=lambda c: abs(c))
            if abs(best) < target_correlation_threshold:
                continue
            findings.append({
                "column": column,
                "check": "target_correlation",
                "score": round(abs(best), 6),
                "correlation": round(best, 6),
                "pearson": None if pd.isna(pearson) else round(float(pearson), 6),
                "rank_correlation": None if rank_corr is None else round(rank_corr, 6),
                "target": target,
                "message": (
                    f"{column!r} is almost perfectly correlated with target {target!r} "
                    f"(max |corr|={abs(best):.4f}); confirm it is available at prediction time."
                ),
            })

    # Stable ordering makes reports and regression tests deterministic.
    findings.sort(key=lambda item: (item["column"], item["check"], item.get("target") or ""))
    return {
        "sample_rows": int(len(sample)),
        "total_rows": int(len(df)),
        "row_order_threshold": float(row_order_threshold),
        "target_correlation_threshold": float(target_correlation_threshold),
        "flagged_columns": sorted({item["column"] for item in findings}),
        "findings": findings,
    }


def format_leakage_report(report: dict[str, Any]) -> str:
    """Format a leakage report for logs or user-facing exploration output."""
    findings = report.get("findings", [])
    if not findings:
        return "Leakage check: no obvious leakage candidates found."
    lines = [f"Leakage check: {len(findings)} finding(s) — review before training:"]
    lines.extend(f"- {finding['message']}" for finding in findings)
    return "\n".join(lines)
