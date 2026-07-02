"""Data loading and feature engineering utilities for PdM projects."""

import os
import numpy as np
import pandas as pd
from typing import Callable


def load_or_cache(name: str, loader: Callable[[], pd.DataFrame], cache_dir: str = "./data") -> pd.DataFrame:
    """Load a DataFrame via `loader`, caching the result as parquet."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"_cache_{name}.parquet")
    if os.path.exists(path):
        print(f"  [cache hit] {name}")
        return pd.read_parquet(path)
    print(f"  [loading] {name}...")
    df = loader()
    df.to_parquet(path, index=False)
    print(f"  [cached] {name} -> {df.shape}")
    return df


def load_eav_chunked(
    bucket: str,
    prefix: str,
    entity_col: str,
    attribute_col: str,
    value_col: str,
    open_parquet_file_fn: Callable,
    get_bucket_region_fn: Callable,
    partition_col: str | None = None,
) -> pd.DataFrame:
    """Load a large EAV telemetry table via per-partition aggregation.

    Processes one partition at a time — keeps memory bounded regardless of total table size.

    Args:
        partition_col: If provided, preserves per-partition granularity instead of combining
            into a single global row per entity. The partition date is extracted from the
            Hive partition directory name and stored in this column. Useful when you need
            per-date stats but don't need the full load_eav_temporal() observation-date alignment.

    Returns:
        If partition_col is None: DataFrame with [entity_col, attribute_col, 'mean', 'std', 'min', 'max', 'count']
            — one row per (entity, attribute) with global stats.
        If partition_col is set: DataFrame with [entity_col, attribute_col, partition_col, 'mean', 'std', 'min', 'max', 'count']
            — one row per (entity, attribute, partition_date).
    """
    import boto3

    s3 = boto3.client("s3", region_name=get_bucket_region_fn(bucket))
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
    partition_prefixes = sorted(p["Prefix"] for p in resp.get("CommonPrefixes", []))
    print(f"  EAV chunked load: {len(partition_prefixes)} partitions to process")

    columns = [entity_col, attribute_col, value_col]
    agg_chunks = []

    for i, part_prefix in enumerate(partition_prefixes):
        resp2 = s3.list_objects_v2(Bucket=bucket, Prefix=part_prefix)
        files = [o["Key"] for o in resp2.get("Contents", []) if o["Key"].endswith(".parquet")]
        if not files:
            continue

        part_dfs = []
        for f in files:
            pf = open_parquet_file_fn(f"{bucket}/{f}")
            for rg in range(pf.metadata.num_row_groups):
                table = pf.read_row_group(rg, columns=columns)
                part_dfs.append(table.to_pandas())

        if not part_dfs:
            continue

        chunk = pd.concat(part_dfs, ignore_index=True)
        chunk[entity_col] = chunk[entity_col].astype(str)
        chunk[value_col] = pd.to_numeric(chunk[value_col], errors="coerce")
        chunk = chunk.dropna(subset=[value_col, attribute_col])

        chunk_agg = chunk.groupby([entity_col, attribute_col])[value_col].agg(
            ["mean", "std", "min", "max", "count"]
        ).reset_index()

        if partition_col:
            # Extract date from partition prefix (e.g. "telemetry/sampled_on=2026-03-02/")
            part_name = part_prefix.rstrip("/").split("/")[-1]
            if "=" in part_name:
                chunk_agg[partition_col] = part_name.split("=", 1)[1]
            else:
                chunk_agg[partition_col] = part_name

        agg_chunks.append(chunk_agg)

        if (i + 1) % 10 == 0:
            print(f"    Processed {i+1}/{len(partition_prefixes)} partitions")

    if not agg_chunks:
        cols = [entity_col, attribute_col, "mean", "std", "min", "max", "count"]
        if partition_col:
            cols.insert(2, partition_col)
        return pd.DataFrame(columns=cols)

    all_agg = pd.concat(agg_chunks, ignore_index=True)

    if partition_col:
        # Return per-partition stats directly (no combining)
        return all_agg

    combined = all_agg.groupby([entity_col, attribute_col]).apply(
        _combine_partition_stats, include_groups=False
    ).reset_index()
    return combined


def _combine_partition_stats(g: pd.DataFrame) -> pd.Series:
    """Combine pre-aggregated partition stats into global stats."""
    total_count = g["count"].sum()
    if total_count == 0:
        return pd.Series({"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0})
    weighted_mean = (g["mean"] * g["count"]).sum() / total_count
    global_max = g["max"].max()
    global_min = g["min"].min()
    within_var = ((g["std"] ** 2) * (g["count"] - 1)).sum() / max(total_count - 1, 1)
    between_var = ((g["mean"] - weighted_mean) ** 2 * g["count"]).sum() / max(total_count - 1, 1)
    global_std = np.sqrt(within_var + between_var)
    return pd.Series({"mean": weighted_mean, "std": global_std, "min": global_min, "max": global_max, "count": total_count})


def load_eav_temporal(
    bucket: str,
    prefix: str,
    entity_col: str,
    attribute_col: str,
    value_col: str,
    observation_dates: list,
    lookback_days: int,
    open_parquet_file_fn: Callable,
    get_bucket_region_fn: Callable,
    partition_date_format: str = "sampled_on=%Y-%m-%d",
) -> pd.DataFrame:
    """Load EAV telemetry with per-observation-date rolling window aggregation.

    Uses a sliding window over partitions — only keeps `lookback_days` worth of
    partition aggregates in memory at any time, evicting old ones as the window
    advances. Memory usage is O(lookback_days × entities × sensors) regardless
    of total partition count.

    Args:
        bucket: S3 bucket name.
        prefix: S3 prefix for partitioned telemetry (e.g. 'telemetry/').
        entity_col: Column identifying the device (e.g. 'device_id').
        attribute_col: Column identifying the sensor (e.g. 'sensor_name').
        value_col: Column with the numeric reading (e.g. 'sample_value').
        observation_dates: List of dates (pd.Timestamp or date-like) to compute features for.
        lookback_days: Number of days before each observation_date to include.
        open_parquet_file_fn: Function to open a parquet file from S3.
        get_bucket_region_fn: Function to get bucket region.
        partition_date_format: strptime format to extract date from partition prefix name.

    Returns:
        DataFrame with [entity_col, '_observation_date', attribute_col, 'mean', 'std', 'min', 'max', 'count']
        — one row per (entity, observation_date, attribute) with stats computed over the lookback window.
    """
    import boto3
    from collections import OrderedDict
    from datetime import datetime

    s3 = boto3.client("s3", region_name=get_bucket_region_fn(bucket))
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
    partition_prefixes = sorted(p["Prefix"] for p in resp.get("CommonPrefixes", []))

    # Parse partition dates from prefix names
    partition_info = []  # [(prefix, date)] sorted by date
    for pp in partition_prefixes:
        part_name = pp.rstrip("/").split("/")[-1]
        try:
            dt = datetime.strptime(part_name, partition_date_format).date()
            partition_info.append((pp, pd.Timestamp(dt)))
        except ValueError:
            continue
    partition_info.sort(key=lambda x: x[1])

    observation_dates = sorted(pd.Timestamp(d) for d in observation_dates)
    print(f"  EAV temporal load: {len(partition_info)} partitions, {len(observation_dates)} obs dates, lookback={lookback_days}d")

    columns = [entity_col, attribute_col, value_col]

    def _load_partition(pp):
        resp2 = s3.list_objects_v2(Bucket=bucket, Prefix=pp)
        files = [o["Key"] for o in resp2.get("Contents", []) if o["Key"].endswith(".parquet")]
        if not files:
            return None
        part_dfs = []
        for f in files:
            pf = open_parquet_file_fn(f"{bucket}/{f}")
            for rg in range(pf.metadata.num_row_groups):
                table = pf.read_row_group(rg, columns=columns)
                part_dfs.append(table.to_pandas())
        if not part_dfs:
            return None
        chunk = pd.concat(part_dfs, ignore_index=True)
        chunk[entity_col] = chunk[entity_col].astype(str)
        chunk[value_col] = pd.to_numeric(chunk[value_col], errors="coerce")
        chunk = chunk.dropna(subset=[value_col, attribute_col])
        return chunk.groupby([entity_col, attribute_col])[value_col].agg(
            ["mean", "std", "min", "max", "count"]
        ).reset_index()

    # Sliding window: cache maps partition_date -> aggregated df
    cache = OrderedDict()  # date -> df (ordered by insertion = chronological)
    part_idx = 0  # pointer into partition_info
    results = []
    loaded_count = 0

    for obs_date in observation_dates:
        window_start = obs_date - pd.Timedelta(days=lookback_days)

        # Load any new partitions up to obs_date
        while part_idx < len(partition_info) and partition_info[part_idx][1] <= obs_date:
            pp, pdate = partition_info[part_idx]
            if pdate not in cache:
                agg = _load_partition(pp)
                if agg is not None:
                    cache[pdate] = agg
                    loaded_count += 1
                    if loaded_count % 10 == 0:
                        print(f"    Loaded {loaded_count}/{len(partition_info)} partitions")
            part_idx += 1

        # Evict partitions outside the window
        evict = [d for d in cache if d <= window_start]
        for d in evict:
            del cache[d]

        # Combine partitions in window
        window_dfs = [df for d, df in cache.items() if d > window_start and d <= obs_date]
        if not window_dfs:
            continue

        window_data = pd.concat(window_dfs, ignore_index=True)
        combined = window_data.groupby([entity_col, attribute_col]).apply(
            _combine_partition_stats, include_groups=False
        ).reset_index()
        combined["_observation_date"] = obs_date
        results.append(combined)

    if not results:
        return pd.DataFrame(columns=[entity_col, "_observation_date", attribute_col, "mean", "std", "min", "max", "count"])

    final = pd.concat(results, ignore_index=True)
    print(f"  EAV temporal result: {final.shape[0]} rows, {final[entity_col].nunique()} entities × {final['_observation_date'].nunique()} dates")
    return final


def pivot_eav(
    df: pd.DataFrame,
    entity_col: str,
    attribute_col: str,
    value_col: str,
    aggs: tuple[str, ...] = ("mean", "std", "max", "min", "count"),
) -> pd.DataFrame:
    """Pivot an Entity-Attribute-Value table to wide format with aggregations."""
    df = df[[entity_col, attribute_col, value_col]].copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])
    if df.empty:
        return pd.DataFrame(columns=[entity_col])
    grouped = df.groupby([entity_col, attribute_col])[value_col].agg(list(aggs))
    grouped.columns = list(aggs)
    grouped = grouped.reset_index()
    pivoted = grouped.pivot_table(index=entity_col, columns=attribute_col, values=list(aggs))
    pivoted.columns = [f"{attr}_{agg}" for agg, attr in pivoted.columns]
    return pivoted.reset_index()


def pivot_precomputed_eav(
    df: pd.DataFrame,
    entity_col: str,
    attribute_col: str,
    stat_cols: tuple[str, ...] = ("mean", "std", "max", "min"),
    min_coverage: float = 0.05,
) -> pd.DataFrame:
    """Pivot pre-aggregated EAV stats into wide format with coverage filtering."""
    # Guard: only use stat_cols that actually exist in the DataFrame
    available_stats = [s for s in stat_cols if s in df.columns]
    if not available_stats:
        raise ValueError(f"None of the requested stat_cols {stat_cols} found in DataFrame columns: {list(df.columns)}")
    if len(available_stats) < len(stat_cols):
        missing = set(stat_cols) - set(available_stats)
        print(f"  ⚠️ pivot_precomputed_eav: stat columns {missing} not found in data, using {available_stats}")

    total_entities = df[entity_col].nunique()
    coverage = df.groupby(attribute_col)[entity_col].nunique() / total_entities
    valid_attrs = coverage[coverage >= min_coverage].index.tolist()
    dropped_attrs = coverage[coverage < min_coverage].index.tolist()
    print(f"  Pivot precomputed: {len(valid_attrs)} attributes pass {min_coverage:.0%} coverage (of {len(coverage)} total)")
    if dropped_attrs:
        print(f"  Dropped {len(dropped_attrs)} low-coverage attributes: {dropped_attrs[:10]}{'...' if len(dropped_attrs) > 10 else ''}")
    filtered = df[df[attribute_col].isin(valid_attrs)]
    pivoted = filtered.pivot_table(index=entity_col, columns=attribute_col, values=available_stats)
    pivoted.columns = [f"{attr}_{stat}" for stat, attr in pivoted.columns]
    return pivoted.reset_index()


def pivot_precomputed_eav_temporal(
    df: pd.DataFrame,
    entity_col: str,
    attribute_col: str,
    date_col: str = "_observation_date",
    stat_cols: tuple[str, ...] = ("mean", "std", "max", "min"),
    min_coverage: float = 0.05,
) -> pd.DataFrame:
    """Pivot pre-aggregated temporal EAV stats into wide format.

    Like `pivot_precomputed_eav` but preserves the observation date,
    producing one row per (entity, observation_date).
    """
    available_stats = [s for s in stat_cols if s in df.columns]
    if not available_stats:
        raise ValueError(f"None of {stat_cols} found in columns: {list(df.columns)}")

    # Coverage: fraction of (entity × date) pairs that have this attribute
    total_keys = df.groupby([entity_col, date_col]).ngroups
    coverage = df.groupby(attribute_col).apply(
        lambda g: g.drop_duplicates(subset=[entity_col, date_col]).shape[0],
        include_groups=False,
    ) / total_keys
    valid_attrs = coverage[coverage >= min_coverage].index.tolist()
    print(f"  Pivot temporal: {len(valid_attrs)} attributes pass {min_coverage:.0%} coverage (of {len(coverage)} total)")
    filtered = df[df[attribute_col].isin(valid_attrs)]
    pivoted = filtered.pivot_table(index=[entity_col, date_col], columns=attribute_col, values=available_stats)
    pivoted.columns = [f"{attr}_{stat}" for stat, attr in pivoted.columns]
    return pivoted.reset_index()


def safe_age_days(dates, reference=None) -> pd.Series:
    """Compute age in days from date values, safely handling various types."""
    if reference is None:
        reference = pd.Timestamp.now()
    parsed = pd.to_datetime(pd.Series(dates), errors="coerce")
    return (reference - parsed).dt.days


def encode_categoricals(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Label-encode categorical columns."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        codes, _ = pd.factorize(df[col])
        df[f"{col}_enc"] = codes
        df = df.drop(columns=[col])
    return df


def booleans_to_int(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert boolean columns to 0/1 integers, safely handling None/NaN."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].map({True: 1, False: 0, None: 0}).fillna(0).astype(int)
    return df


def deduplicate_on(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Deduplicate a DataFrame on a key column, keeping the first occurrence."""
    n_before = len(df)
    df = df.drop_duplicates(subset=[key], keep="first")
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  Deduplicated on '{key}': dropped {n_dropped} rows ({n_before} → {len(df)})")
    return df


def load_and_prepare_metadata(
    id_col: str,
    load_fn: Callable,
    keep_cols: list[str] | None = None,
    date_cols_for_age: list[str] | None = None,
    reference_date=None,
    # Deprecated — kept for backward compat, ignored internally
    bucket: str | None = None,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Load a flat metadata table, deduplicate, cast ID, convert booleans, compute ages.

    Standard pattern for preparing device/asset master data before joining.

    Args:
        id_col: Primary key column (will be cast to str and deduplicated).
        load_fn: No-arg callable that returns a DataFrame,
            e.g. ``lambda: load_all_flat_parquet(bucket, prefix)``.
        keep_cols: If provided, select only these columns (id_col always included).
        date_cols_for_age: Date columns to convert to ``*_age_days`` features.
        reference_date: Reference date for age computation (default: now).
        bucket: DEPRECATED — ignored. Kept for backward compatibility.
        prefix: DEPRECATED — ignored. Kept for backward compatibility.

    Returns:
        Cleaned DataFrame with: id as str, booleans as 0/1, dates as age_days.
    """
    df = load_fn()
    df[id_col] = df[id_col].astype(str)
    df = deduplicate_on(df, id_col)

    if keep_cols:
        cols = [id_col] + [c for c in keep_cols if c in df.columns and c != id_col]
        df = df[cols]

    # Convert booleans
    bool_cols = [c for c in df.columns if df[c].dtype == "bool"]
    if bool_cols:
        df = booleans_to_int(df, bool_cols)

    # Convert date columns to age in days
    if date_cols_for_age:
        for col in date_cols_for_age:
            if col in df.columns:
                age_col = col.replace("_date", "").replace("_at", "") + "_age_days"
                df[age_col] = safe_age_days(df[col], reference=reference_date)
                df = df.drop(columns=[col])

    return df


# --- Dataset construction utilities ---


def sanitize_label_name(issue_title: str) -> str:
    """Convert an issue title to a valid label_* column name."""
    name = issue_title.replace(" ", "_").replace("(", "").replace(")", "")
    name = name.replace("<", "lt").replace("%", "pct").replace("/", "_")
    name = name.replace(",", "").replace("'", "").replace('"', "")
    return f"label_{name}"


def build_multilabel_matrix(
    devices: list,
    observation_dates: list,
    events_df: pd.DataFrame | None = None,
    device_col: str = "",
    date_col: str = "",
    issue_col: str = "",
    kept_issues: list[str] | None = None,
    horizon_days: int = 7,
    # Deprecated alias — kept for backward compat
    health_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a forward-looking multi-label matrix.

    For each (device, observation_date), set label_X = 1 if the device has
    issue X during [obs_date + 1, obs_date + horizon_days].

    Works with any event source: health monitoring tables, support cases/tickets,
    alarm logs, or maintenance records. The events_df just needs a device ID,
    a date, and an issue/category column.

    Uses vectorized merge instead of nested loops for scalability.

    Args:
        devices: List of device IDs to include.
        observation_dates: List of observation timestamps.
        events_df: DataFrame with event records (cases, tickets, health events, alarms).
        device_col: Column name for device identifier in events_df.
        date_col: Column name for date in events_df.
        issue_col: Column name for issue/failure type in events_df.
        kept_issues: List of issue titles to create labels for.
        horizon_days: Forward-looking window size in days.
        health_df: DEPRECATED — use events_df instead. Kept for backward compatibility.

    Returns:
        DataFrame with columns [device_col, '_observation_date', 'label_*' per issue].
    """
    # Support deprecated health_df kwarg
    if events_df is None and health_df is not None:
        events_df = health_df
    if events_df is None:
        raise ValueError("events_df (or deprecated health_df) must be provided")

    # Build observation grid
    grid = pd.DataFrame(
        [(d, pd.Timestamp(obs)) for obs in observation_dates for d in devices],
        columns=[device_col, "_observation_date"],
    )
    grid[device_col] = grid[device_col].astype(str)

    # Prepare event data
    h = events_df[[device_col, date_col, issue_col]].copy()
    h[device_col] = h[device_col].astype(str)
    h[date_col] = pd.to_datetime(h[date_col])
    h = h[h[issue_col].isin(kept_issues)]

    # Merge grid with events on device, then filter by horizon window
    merged = grid.merge(h, on=device_col, how="left")
    merged["_days_ahead"] = (merged[date_col] - merged["_observation_date"]).dt.days
    merged = merged[(merged["_days_ahead"] >= 1) & (merged["_days_ahead"] <= horizon_days)]

    # Pivot to get one column per issue
    label_col_names = [sanitize_label_name(issue) for issue in kept_issues]
    for issue, col_name in zip(kept_issues, label_col_names):
        hits = merged[merged[issue_col] == issue].drop_duplicates(
            subset=[device_col, "_observation_date"]
        )[[device_col, "_observation_date"]]
        hits[col_name] = 1
        grid = grid.merge(hits, on=[device_col, "_observation_date"], how="left")
        grid[col_name] = grid[col_name].fillna(0).astype(int)

    return grid


def temporal_split(
    df: pd.DataFrame,
    date_col: str = "_observation_date",
    train_frac: float = 0.8,
    unit_col: str | None = None,
    cutoff_date: str | None = None,
    horizon_days: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataset temporally (no future leakage).

    For classification/multi-label: splits by observation date quantile (or explicit cutoff_date).
    For RUL (unit_col provided): splits by unit, assigning earlier units to train.

    Args:
        df: Dataset with a date column.
        date_col: Name of the temporal column.
        train_frac: Fraction of data for training (used if cutoff_date not provided).
        unit_col: If set, split by unit instead of date (for RUL formulation).
        cutoff_date: Explicit cutoff date string (e.g. '2024-06-01'). Overrides train_frac.
        horizon_days: If set, enforces a gap between train and test to prevent label
            leakage when labels use a forward-looking window. Test starts at
            cutoff + horizon_days instead of cutoff + 1 day.

    Returns (train_df, test_df) with date_col dropped.
    """
    if unit_col and unit_col in df.columns:
        # RUL: split by unit based on last observation
        unit_last = df.groupby(unit_col)[date_col].max().sort_values()
        cutoff_idx = int(len(unit_last) * train_frac)
        train_units = set(unit_last.index[:cutoff_idx])
        train = df[df[unit_col].isin(train_units)]
        test = df[~df[unit_col].isin(train_units)]
    else:
        # Classification: split by date
        df[date_col] = pd.to_datetime(df[date_col])
        if cutoff_date is not None:
            cutoff = pd.Timestamp(cutoff_date)
        else:
            cutoff = df[date_col].quantile(train_frac)
        train = df[df[date_col] <= cutoff]
        if horizon_days:
            test_start = cutoff + pd.Timedelta(days=horizon_days)
            test = df[df[date_col] > test_start]
            print(f"  Temporal split: cutoff={cutoff.date()}, gap={horizon_days}d, test_start={test_start.date()}, train={len(train)}, test={len(test)}")
        else:
            test = df[df[date_col] > cutoff]
            print(f"  Temporal split: cutoff={cutoff.date()}, train={len(train)}, test={len(test)}")

    # Drop internal date column
    drop = [c for c in [date_col] if c in train.columns]
    return train.drop(columns=drop), test.drop(columns=drop)


def drop_zero_variance(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    exclude_prefixes: tuple = ("label_",),
    verbose: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Drop zero-variance features from train and align test to match.

    Detects zero-variance columns in train, drops them, then applies the same
    column set to test (filling missing with 0, dropping extras).

    Args:
        train: Training DataFrame (variance is computed on this).
        test: Optional test DataFrame. If provided, aligned to train's surviving columns.
        exclude_prefixes: Column prefixes to never drop (labels, targets).
        verbose: Print which features are dropped.

    Returns:
        If test is None: filtered train DataFrame.
        If test is provided: tuple of (filtered_train, aligned_test) with identical columns.
    """
    feature_cols = [
        c for c in train.columns
        if not any(c.startswith(p) for p in exclude_prefixes)
        and c not in ("RUL", "machine_failure", "duration", "event", "unit_id", "cycle", "_observation_date")
    ]
    zero_var = [c for c in feature_cols if train[c].nunique() <= 1]
    if zero_var and verbose:
        print(f"  Dropping {len(zero_var)} zero-variance features: {zero_var[:5]}{'...' if len(zero_var) > 5 else ''}")

    train_out = train.drop(columns=zero_var) if zero_var else train

    if test is None:
        return train_out

    # Align test to train's columns
    keep_cols = train_out.columns.tolist()
    for c in keep_cols:
        if c not in test.columns:
            test[c] = 0
    test_out = test[keep_cols]
    return train_out, test_out


def save_dataset(df: pd.DataFrame, output_dir: str = "./data", date_col: str = "_observation_date",
                 train_frac: float = 0.8, unit_col: str | None = None):
    """Split and save a dataset to train.csv, test.csv, dataset.csv.

    Handles temporal splitting, zero-variance removal, and NaN filling.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = drop_zero_variance(df)
    df = df.fillna(0)

    train, test = temporal_split(df, date_col=date_col, train_frac=train_frac, unit_col=unit_col)

    full = pd.concat([train, test], ignore_index=True)
    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    full.to_csv(os.path.join(output_dir, "dataset.csv"), index=False)

    print(f"  Saved: train={train.shape}, test={test.shape}, dataset={full.shape}")
    return train, test


def align_to_model(df: pd.DataFrame, metadata_path: str = "./model/metadata.json") -> pd.DataFrame:
    """Align a DataFrame's columns to match the trained model's expected features.

    Reindexes columns to the exact order in metadata.json, adding missing columns
    as 0 and dropping extra columns. Prevents silent feature misalignment at inference.

    Args:
        df: DataFrame with feature columns (e.g., from runtime feature engineering).
        metadata_path: Path to the model's metadata.json file.

    Returns:
        DataFrame with columns matching the trained model exactly.
    """
    import json
    with open(metadata_path) as f:
        metadata = json.load(f)
    expected = metadata["feature_names"]
    missing = set(expected) - set(df.columns)
    if missing:
        print(f"  ⚠️ align_to_model: {len(missing)} features missing, filling with 0: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
    extra = set(df.columns) - set(expected)
    if extra:
        print(f"  align_to_model: dropping {len(extra)} extra columns not in model")
    return df.reindex(columns=expected, fill_value=0)


def window_features(
    df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    value_cols: list[str],
    windows: list[int] = (7, 14, 30),
) -> pd.DataFrame:
    """Compute rolling window statistics per entity for temporal feature engineering.

    For each entity and window size, computes: mean, std, slope (linear trend),
    range, and deviation (last value minus window mean).

    Args:
        df: DataFrame sorted by [entity_col, time_col].
        entity_col: Column identifying the device/unit.
        time_col: Column with timestamps or cycle numbers (used for ordering).
        value_cols: Sensor/feature columns to compute rolling stats on.
        windows: List of window sizes (in rows).

    Returns:
        DataFrame with one row per entity (last observation), columns named
        `{col}_w{window}_{stat}`.
    """
    df = df.sort_values([entity_col, time_col])
    results = []
    for entity, group in df.groupby(entity_col):
        row = {entity_col: entity}
        for col in value_cols:
            series = group[col].astype(float)
            for w in windows:
                tail = series.tail(w)
                if len(tail) < 2:
                    continue
                prefix = f"{col}_w{w}"
                row[f"{prefix}_mean"] = tail.mean()
                row[f"{prefix}_std"] = tail.std()
                row[f"{prefix}_range"] = tail.max() - tail.min()
                row[f"{prefix}_dev"] = float(series.iloc[-1]) - tail.mean()
                # slope via simple linear regression
                x = np.arange(len(tail))
                row[f"{prefix}_slope"] = np.polyfit(x, tail.values, 1)[0] if tail.notna().all() else 0.0
        results.append(row)
    return pd.DataFrame(results)


def health_indicator(df: pd.DataFrame, feature_cols: list[str], label_col: str) -> pd.DataFrame:
    """Score features by monotonicity and trendability relative to the label.

    Returns a DataFrame with columns [feature, monotonicity, trendability, score]
    sorted by composite score descending. Use to filter features that consistently
    degrade toward failure.

    Args:
        df: Dataset with features and label column.
        feature_cols: Columns to evaluate.
        label_col: Binary label or continuous RUL column.

    Returns:
        DataFrame with per-feature health indicator scores.
    """
    from scipy.stats import spearmanr
    results = []
    label = df[label_col].values
    for col in feature_cols:
        vals = df[col].fillna(0).values
        if np.std(vals) == 0:
            results.append({"feature": col, "monotonicity": 0.0, "trendability": 0.0, "score": 0.0})
            continue
        # Trendability: absolute Spearman correlation with label
        corr, _ = spearmanr(vals, label)
        trendability = abs(corr) if not np.isnan(corr) else 0.0
        # Monotonicity: fraction of consecutive differences that share the same sign
        diffs = np.diff(vals)
        if len(diffs) > 0:
            monotonicity = abs(np.sum(np.sign(diffs))) / len(diffs)
        else:
            monotonicity = 0.0
        score = (trendability + monotonicity) / 2
        results.append({"feature": col, "monotonicity": round(monotonicity, 4),
                        "trendability": round(trendability, 4), "score": round(score, 4)})
    return pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)


def freq_features(signal: np.ndarray, sampling_rate: float = 1.0) -> dict:
    """Compute frequency-domain features from a 1D signal.

    Useful for vibration, current, and acoustic sensor data. Requires scipy.

    Args:
        signal: 1D numpy array of sensor readings.
        sampling_rate: Sampling frequency in Hz (default 1.0 for normalized).

    Returns:
        Dict with: dominant_freq, spectral_centroid, spectral_entropy,
        band_power_low, band_power_mid, band_power_high, rms.
    """
    from scipy.fft import rfft, rfftfreq
    from scipy.signal import welch

    n = len(signal)
    if n < 4:
        return {"dominant_freq": 0, "spectral_centroid": 0, "spectral_entropy": 0,
                "band_power_low": 0, "band_power_mid": 0, "band_power_high": 0, "rms": 0}

    fft_vals = np.abs(rfft(signal))
    freqs = rfftfreq(n, d=1.0 / sampling_rate)

    # Dominant frequency
    dominant_freq = float(freqs[np.argmax(fft_vals[1:]) + 1]) if len(fft_vals) > 1 else 0.0

    # Spectral centroid
    power = fft_vals ** 2
    total_power = power.sum()
    spectral_centroid = float((freqs * power).sum() / total_power) if total_power > 0 else 0.0

    # Spectral entropy
    p = power / total_power if total_power > 0 else np.zeros_like(power)
    p = p[p > 0]
    spectral_entropy = float(-np.sum(p * np.log2(p))) if len(p) > 0 else 0.0

    # Band powers (low/mid/high thirds of spectrum)
    n_bins = len(freqs)
    third = n_bins // 3
    band_power_low = float(power[:third].sum())
    band_power_mid = float(power[third:2*third].sum())
    band_power_high = float(power[2*third:].sum())

    # RMS
    rms = float(np.sqrt(np.mean(signal ** 2)))

    return {
        "dominant_freq": round(dominant_freq, 6),
        "spectral_centroid": round(spectral_centroid, 6),
        "spectral_entropy": round(spectral_entropy, 6),
        "band_power_low": round(band_power_low, 6),
        "band_power_mid": round(band_power_mid, 6),
        "band_power_high": round(band_power_high, 6),
        "rms": round(rms, 6),
    }


# --- Explainability utilities ---


def feature_contributions(
    predict_fn: Callable,
    row: pd.DataFrame,
    feature_names: list[str],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Compute per-feature contribution via perturbation (zeroing out).

    For a single sample, measures how much each feature contributes to the
    prediction by setting it to 0 and observing the probability change.

    Args:
        predict_fn: Callable that takes a DataFrame row and returns a probability
            (float). For AutoGluon: ``lambda x: predictor.predict_proba(x)[1].values[0]``
        row: Single-row DataFrame with feature values.
        feature_names: Features to perturb (e.g., top-N from feature importance).
        top_k: Number of top contributors to return.

    Returns:
        List of (feature_name, delta) tuples sorted by |delta| descending.
        Positive delta means the feature increases the prediction probability.
    """
    base_prob = predict_fn(row)
    contribs = {}
    for col in feature_names:
        if col in row.columns:
            perturbed = row.copy()
            perturbed[col] = 0.0
            new_prob = predict_fn(perturbed)
            contribs[col] = float(base_prob - new_prob)
    return sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
