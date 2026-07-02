import pyarrow.parquet as pq
import pyarrow.fs as pafs
import boto3


def _get_bucket_region(bucket: str) -> str:
    """Detect the region of an S3 bucket."""
    loc = boto3.client("s3").get_bucket_location(Bucket=bucket)["LocationConstraint"]
    return loc or "us-east-1"  # None means us-east-1


def _serialize(val):
    """Convert value to JSON-safe type."""
    import numpy as np
    import pandas as pd

    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, (pd.Timestamp, np.datetime64)):
        return str(val)
    if isinstance(val, bytes):
        return val.hex()
    return val


def open_parquet_file(s3_uri: str):
    """Open a parquet file on S3 and return a PyArrow ParquetFile object for reading.

    Use this to load data into a pandas DataFrame, e.g.:
        pf = open_parquet_file('s3://bucket/path/file.parquet')
        df = pf.read().to_pandas()

    Args:
        s3_uri: S3 path in format 'bucket/key' or 's3://bucket/key'.

    Returns:
        pyarrow.parquet.ParquetFile ready for .read(), .read_row_group(), etc.
    """
    # Normalize URI and detect region
    path = s3_uri.removeprefix("s3://")
    bucket = path.split("/")[0]
    fs = pafs.S3FileSystem(region=_get_bucket_region(bucket))

    return pq.ParquetFile(fs.open_input_file(path))


def explore_schema(s3_uri: str, sample_size: int = 100_000, max_value_len: int = 80,
                   min_coverage: float = 0.0, max_columns: int | None = None,
                   prefix: str | None = None) -> dict:
    """Return a concise schema summary suitable for LLM context windows.

    Unlike explore_parquet_file(), this function:
    - Truncates string values in value_counts to max_value_len chars
    - Limits value_counts to top 15 (not 60)
    - Skips value_counts entirely for columns with >500 unique values (high-cardinality)
    - Reports total row count from parquet metadata (all row groups)

    Args:
        s3_uri: S3 path in format 'bucket/key' or 's3://bucket/key'.
            When `prefix` is provided, this is interpreted as just the bucket name
            (e.g. 's3://my-bucket' or 'my-bucket').
        sample_size: Max rows to analyze (default 100K — enough for distributions).
        max_value_len: Truncate string representations longer than this.
        min_coverage: Only include columns where non-null fraction >= this value (0.0–1.0).
        max_columns: Limit output to N columns, ranked by coverage then cardinality.
        prefix: If provided, treat as a flat multi-file table prefix. Reads schema from
            the largest file and samples rows across multiple files for better coverage.

    Returns:
        Dict with s3_uri, total_rows, num_columns, and per-column summary.
    """
    import pandas as pd

    if prefix is not None:
        # Multi-file mode: resolve bucket, find files, pick largest for schema
        bucket_name = s3_uri.removeprefix("s3://").rstrip("/").split("/")[0]
        s3 = boto3.client("s3", region_name=_get_bucket_region(bucket_name))
        resp = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        files = [(o["Key"], o["Size"]) for o in resp.get("Contents", [])
                 if o["Key"].endswith(".parquet")]
        if not files:
            return {"s3_uri": f"s3://{bucket_name}/{prefix}", "total_rows": 0,
                    "num_columns": 0, "sample_rows": 0, "columns": {}}
        files.sort(key=lambda x: x[1], reverse=True)
        largest_key = files[0][0]
        s3_uri = f"s3://{bucket_name}/{largest_key}"

        # Compute total rows across all files from metadata
        total_rows = 0
        for key, _ in files:
            pf_meta = open_parquet_file(f"{bucket_name}/{key}")
            total_rows += sum(pf_meta.metadata.row_group(i).num_rows
                             for i in range(pf_meta.metadata.num_row_groups))

        # Sample rows from multiple files for better distribution coverage
        rows_per_file = max(sample_size // min(len(files), 5), 1000)
        sample_dfs = []
        for key, _ in files[:5]:
            pf_tmp = open_parquet_file(f"{bucket_name}/{key}")
            tbl = pf_tmp.read_row_group(0).slice(0, rows_per_file)
            sample_dfs.append(tbl.to_pandas())
        sample_df = pd.concat(sample_dfs, ignore_index=True).head(sample_size)

        # Use largest file for schema reference
        pf = open_parquet_file(s3_uri)
        metadata = pf.metadata
    else:
        pf = open_parquet_file(s3_uri)
        metadata = pf.metadata
        total_rows = sum(metadata.row_group(i).num_rows for i in range(metadata.num_row_groups))
        sample_table = pf.read_row_group(0, use_pandas_metadata=True).slice(0, sample_size)
        sample_df = sample_table.to_pandas()

    columns = {}
    for field in pf.schema_arrow:
        col_name = field.name
        col_data = sample_df[col_name] if col_name in sample_df.columns else None
        col_info = {"dtype": str(field.type)}

        if col_data is not None:
            n_null = int(col_data.isna().sum())
            n_unique = int(col_data.nunique())
            col_info["nulls"] = n_null
            col_info["unique"] = n_unique

            # Skip value_counts for high-cardinality or text-heavy columns
            if n_unique <= 500:
                try:
                    vc = col_data.dropna().value_counts().head(15)
                    col_info["top_values"] = {
                        str(_serialize(k))[:max_value_len]: int(v)
                        for k, v in vc.items()
                    }
                except TypeError:
                    pass

            # Stats for numeric
            if col_data.dtype.kind in ("i", "f"):
                non_null = col_data.dropna()
                if len(non_null):
                    col_info["min"] = _serialize(non_null.min())
                    col_info["max"] = _serialize(non_null.max())

        columns[col_name] = col_info

    # Apply coverage and column-count filters
    n_rows = len(sample_df)
    if min_coverage > 0.0 and n_rows > 0:
        columns = {k: v for k, v in columns.items()
                   if (n_rows - v.get("nulls", 0)) / n_rows >= min_coverage}
    if max_columns is not None and len(columns) > max_columns:
        ranked = sorted(columns.items(),
                        key=lambda kv: ((n_rows - kv[1].get("nulls", 0)) / max(n_rows, 1),
                                        kv[1].get("unique", 0)),
                        reverse=True)
        columns = dict(ranked[:max_columns])

    result = {
        "s3_uri": s3_uri,
        "total_rows": total_rows,
        "num_columns": metadata.num_columns,
        "sample_rows": len(sample_df),
        "columns": columns,
    }

    # Detect Hive partition columns from the S3 path (e.g., "key=value/" patterns)
    import re
    partition_matches = re.findall(r'/([a-zA-Z_][a-zA-Z0-9_]*)=([^/]+)/', s3_uri)
    if partition_matches:
        result["hive_partitions"] = {
            col: {"sample_value": val} for col, val in partition_matches
        }

    return result


def explore_table_summary(bucket: str, prefix: str) -> dict:
    """Get total row count and file count for a flat (non-partitioned) table.

    Reads only parquet metadata (no data loaded) — fast even for many files.

    Args:
        bucket: S3 bucket name.
        prefix: S3 prefix (e.g. 'device_health/').

    Returns:
        Dict with file_count, total_rows, and list of file keys.
    """
    s3 = boto3.client("s3", region_name=_get_bucket_region(bucket))
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    files = [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".parquet")]

    total_rows = 0
    for f in files:
        pf = open_parquet_file(f"{bucket}/{f}")
        total_rows += sum(
            pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)
        )

    return {"prefix": prefix, "file_count": len(files), "total_rows": total_rows, "files": files}


def validate_join(
    bucket: str,
    left_prefix: str,
    right_prefix: str,
    left_key: str,
    right_key: str,
    left_partitioned: bool = False,
    right_partitioned: bool = False,
    max_partitions: int = 1,
) -> dict:
    """Validate a join between two S3 tables by checking key overlap.

    Loads only the join key columns (minimal I/O) and reports overlap statistics.

    Args:
        bucket: S3 bucket name.
        left_prefix: Prefix for left table (e.g. 'telemetry/').
        right_prefix: Prefix for right table (e.g. 'device_master/').
        left_key: Column name in left table.
        right_key: Column name in right table.
        left_partitioned: If True, use load_partitioned_parquet for left table.
        right_partitioned: If True, use load_partitioned_parquet for right table.
        max_partitions: Partitions to sample (applies to whichever side is partitioned).

    Returns:
        Dict with left_unique, right_unique, overlap, pct_of_left, pct_of_right,
        left_dtype, right_dtype.
    """
    if left_partitioned:
        left_df = load_partitioned_parquet(
            bucket, left_prefix, columns=[left_key], max_partitions=max_partitions
        )
    else:
        left_df = load_all_flat_parquet(bucket, left_prefix, columns=[left_key])

    if right_partitioned:
        right_df = load_partitioned_parquet(
            bucket, right_prefix, columns=[right_key], max_partitions=max_partitions
        )
    else:
        right_df = load_all_flat_parquet(bucket, right_prefix, columns=[right_key])

    left_ids = set(left_df[left_key].astype(str))
    right_ids = set(right_df[right_key].astype(str))
    overlap = left_ids & right_ids

    return {
        "left_key": f"{left_prefix}.{left_key}",
        "right_key": f"{right_prefix}.{right_key}",
        "left_unique": len(left_ids),
        "right_unique": len(right_ids),
        "overlap": len(overlap),
        "pct_of_left": round(len(overlap) / max(len(left_ids), 1) * 100, 1),
        "pct_of_right": round(len(overlap) / max(len(right_ids), 1) * 100, 1),
        "left_dtype": str(left_df[left_key].dtype),
        "right_dtype": str(right_df[right_key].dtype),
    }


def discover_join_keys(
    bucket: str,
    prefixes: list[str],
    partitioned: list[str] | None = None,
    max_partitions: int = 1,
) -> list[dict]:
    """Auto-detect candidate join keys across tables by finding shared column names and checking overlap.

    For every pair of tables, finds columns that share the same name, loads only those columns,
    casts both sides to string, and reports overlap %. Useful as a first pass before validate_join().

    Args:
        bucket: S3 bucket name.
        prefixes: List of table prefixes (e.g. ['telemetry/', 'cases/', 'device_master/']).
        partitioned: List of prefixes that are Hive-partitioned (subset of prefixes).
        max_partitions: Partitions to sample from partitioned tables.

    Returns:
        List of dicts with left_table, right_table, column, left_unique, right_unique,
        overlap, pct_of_left, pct_of_right. Sorted by overlap descending.
    """
    import itertools

    partitioned = set(partitioned or [])

    # Discover column names per table (read one file's schema only — no data)
    table_columns: dict[str, set[str]] = {}
    s3 = boto3.client("s3", region_name=_get_bucket_region(bucket))
    for prefix in prefixes:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        files = [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".parquet")]
        if files:
            pf = open_parquet_file(f"{bucket}/{files[0]}")
            table_columns[prefix] = {f.name for f in pf.schema_arrow}

    results = []
    for left, right in itertools.combinations(prefixes, 2):
        if left not in table_columns or right not in table_columns:
            continue
        shared = table_columns[left] & table_columns[right]
        for col in shared:
            try:
                r = validate_join(
                    bucket, left, right, col, col,
                    left_partitioned=(left in partitioned),
                    right_partitioned=(right in partitioned),
                    max_partitions=max_partitions,
                )
                left_dt = r["left_dtype"]
                right_dt = r["right_dtype"]
                results.append({
                    "left_table": left,
                    "right_table": right,
                    "column": col,
                    "left_unique": r["left_unique"],
                    "right_unique": r["right_unique"],
                    "overlap": r["overlap"],
                    "pct_of_left": r["pct_of_left"],
                    "pct_of_right": r["pct_of_right"],
                    "dtypes": (left_dt, right_dt),
                    "type_mismatch": left_dt != right_dt,
                })
            except Exception:
                continue

    results.sort(key=lambda x: x["overlap"], reverse=True)
    return results


def list_eav_attributes(
    bucket: str,
    prefix: str,
    attribute_col: str = "sensor_name",
    group_col: str | None = None,
    partitioned: bool = True,
    max_partitions: int = 1,
) -> dict[str, list[str]] | list[str]:
    """List distinct attribute values from an EAV table without loading full data.

    Uses column projection to read only the attribute (and optional grouping) column,
    minimizing I/O on large tables.

    Args:
        bucket: S3 bucket name.
        prefix: Table prefix (e.g. 'telemetry/').
        attribute_col: Column containing attribute names (e.g. 'sensor_name').
        group_col: Optional grouping column (e.g. 'sensor_category'). If provided,
            returns attributes grouped by this column.
        partitioned: Whether the table is Hive-partitioned.
        max_partitions: Partitions to sample (for partitioned tables).

    Returns:
        - If group_col is None: list of unique attribute values (sorted).
        - If group_col is provided: dict mapping group_name → sorted list of attributes.
          Example: {"M2W_FW": ["ac_current_rms_l1", "temp_l1", ...], "SYSTEM": ["cpu_temp", ...]}
    """
    cols = [attribute_col]
    if group_col:
        cols.append(group_col)

    if partitioned:
        df = load_partitioned_parquet(bucket, prefix, columns=cols, max_partitions=max_partitions)
    else:
        df = load_all_flat_parquet(bucket, prefix, columns=cols)

    if group_col:
        grouped = df.dropna(subset=[attribute_col]).groupby(group_col)[attribute_col].apply(
            lambda x: sorted(x.unique().tolist())
        ).to_dict()
        return grouped

    return sorted(df[attribute_col].dropna().unique().tolist())


def discover_cross_name_joins(
    bucket: str,
    prefixes: list[str],
    partitioned: list[str] | None = None,
    max_partitions: int = 1,
    min_overlap_pct: float = 10.0,
    sample_size: int = 50_000,
) -> list[dict]:
    """Find join candidates where column NAMES differ but VALUES overlap.

    For each pair of tables, samples string/integer columns and checks whether
    their value sets intersect above a threshold — regardless of column name.
    Useful for detecting joins like 'serial_number' ↔ 'device_id'.

    Args:
        bucket: S3 bucket name.
        prefixes: List of table prefixes.
        partitioned: List of prefixes that are Hive-partitioned.
        max_partitions: Partitions to sample from partitioned tables.
        min_overlap_pct: Minimum overlap % (of the smaller side) to report.
        sample_size: Max rows to sample per table for value comparison.

    Returns:
        List of dicts with left_table, right_table, left_column, right_column,
        overlap, pct_of_left, pct_of_right, left_dtype, right_dtype.
        Sorted by overlap descending.
    """
    import itertools
    import pandas as pd

    partitioned = set(partitioned or [])

    def _load_sample(prefix):
        if prefix in partitioned:
            return load_partitioned_parquet(bucket, prefix, max_partitions=1)
        else:
            df = load_all_flat_parquet(bucket, prefix)
            return df.head(sample_size)

    # Load samples and extract candidate columns (string/int, cardinality between 2 and 80% of rows)
    table_data: dict[str, dict[str, set]] = {}
    table_dtypes: dict[str, dict[str, str]] = {}
    for prefix in prefixes:
        try:
            df = _load_sample(prefix)
        except Exception:
            continue
        cols = {}
        dtypes = {}
        for col in df.columns:
            if df[col].dtype.kind not in ("i", "u", "O", "U"):
                continue
            nunique = df[col].nunique()
            if nunique < 2 or nunique > len(df) * 0.8:
                continue
            vals = set(df[col].dropna().astype(str).head(sample_size))
            if len(vals) >= 2:
                cols[col] = vals
                dtypes[col] = str(df[col].dtype)
        table_data[prefix] = cols
        table_dtypes[prefix] = dtypes

    results = []
    for left, right in itertools.combinations(prefixes, 2):
        if left not in table_data or right not in table_data:
            continue
        for lcol, lvals in table_data[left].items():
            for rcol, rvals in table_data[right].items():
                if lcol == rcol:
                    continue  # same-name joins handled by discover_join_keys
                overlap = lvals & rvals
                if not overlap:
                    continue
                pct_left = len(overlap) / max(len(lvals), 1) * 100
                pct_right = len(overlap) / max(len(rvals), 1) * 100
                if max(pct_left, pct_right) < min_overlap_pct:
                    continue
                results.append({
                    "left_table": left,
                    "right_table": right,
                    "left_column": lcol,
                    "right_column": rcol,
                    "overlap": len(overlap),
                    "pct_of_left": round(pct_left, 1),
                    "pct_of_right": round(pct_right, 1),
                    "left_dtype": table_dtypes[left][lcol],
                    "right_dtype": table_dtypes[right][rcol],
                })

    results.sort(key=lambda x: x["overlap"], reverse=True)
    return results


def explore_parquet_file(s3_uri: str, sample_size: int = 3_000_000, max_unique: int = 60) -> dict:
    """Explore a parquet file on S3 and return metadata + column documentation.

    Uses parquet metadata (no full scan) and reads the first row group. Since
    parquet I/O reads the full row group regardless of slice size, we sample as
    many rows as possible without adding more than ~3s of processing overhead.

    Args:
        s3_uri: S3 path in format 'bucket/key' or 's3://bucket/key'.
        sample_size: Max rows to analyze from the first row group (default 3M).
        max_unique: Max unique values to report in value_counts (default 60).

    Returns:
        Dictionary with file metadata, schema info, and per-column documentation
        including dtype, nullability, value_counts (Counter), and basic stats.
    """
    # Read metadata without loading data
    pf = open_parquet_file(s3_uri)
    metadata = pf.metadata

    # Read a small sample (first row group, limited rows)
    sample_table = pf.read_row_group(0, use_pandas_metadata=True).slice(0, sample_size)
    sample_df = sample_table.to_pandas()

    # Build column documentation
    schema = pf.schema_arrow
    columns = {}
    for field in schema:
        col_name = field.name
        col_data = sample_df[col_name] if col_name in sample_df.columns else None

        col_info = {
            "dtype": str(field.type),
            "nullable": field.nullable,
        }

        if col_data is not None:
            non_null = col_data.dropna()
            col_info["null_count_in_sample"] = int(col_data.isna().sum())

            # Value counts as a Counter dict
            try:
                vc = non_null.value_counts()
                col_info["value_counts"] = {str(_serialize(k)): int(v) for k, v in vc.head(max_unique).items()}
                col_info["unique_in_sample"] = int(non_null.nunique())
            except TypeError:
                # unhashable types (e.g. list columns)
                col_info["value_counts"] = {}
                col_info["unique_in_sample"] = 0

            # Basic stats for numeric columns
            if col_data.dtype.kind in ("i", "f"):
                col_info["min"] = _serialize(non_null.min()) if len(non_null) else None
                col_info["max"] = _serialize(non_null.max()) if len(non_null) else None

        columns[col_name] = col_info

    return {
        "s3_uri": s3_uri,
        "num_rows": metadata.num_rows,
        "num_columns": metadata.num_columns,
        "columns": columns,
    }


def _insert(tree: list, parts: list):
    """Insert a path (split into parts) into the nested tree structure."""
    if len(parts) == 1:
        tree.append(parts[0])
        return
    folder_name = parts[0]
    for item in tree:
        if isinstance(item, dict) and folder_name in item:
            _insert(item[folder_name], parts[1:])
            return
    new_list = []
    tree.append({folder_name: new_list})
    _insert(new_list, parts[1:])


def explore_bucket(bucket: str, extension: str = None) -> dict:
    """Return the file tree of an S3 bucket as a nested dict/list structure.

    Args:
        bucket: S3 bucket name.
        extension: If provided, only include files with this extension (e.g. '.parquet').

    Returns:
        Dict with bucket name as key and a list of folder dicts / file strings as value.
    """
    s3 = boto3.client("s3", region_name=_get_bucket_region(bucket))
    tree = []
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if extension and not key.endswith(extension):
                continue
            parts = key.split("/")
            _insert(tree, parts)

    return {bucket: tree}


def load_partitioned_parquet(
    bucket: str,
    prefix: str,
    columns: list[str] | None = None,
    max_partitions: int | None = None,
    max_rows_per_partition: int | None = None,
    partition_filter: str | None = None,
) -> "pd.DataFrame":
    """Load a Hive-partitioned parquet dataset from S3 with column pruning.

    Designed for large IoT/telemetry tables partitioned by date (e.g. sampled_on=YYYY-MM-DD/).
    Reads ALL files in each partition to ensure no data loss.

    Args:
        bucket: S3 bucket name.
        prefix: S3 prefix to the partitioned table (e.g. 'telemetry/').
        columns: List of columns to read. None = all columns.
        max_partitions: Maximum number of partitions to load (sorted ascending). None = all.
        max_rows_per_partition: Max rows to read from each partition file. None = all rows.
        partition_filter: Optional substring to filter partition prefixes (e.g. '2026-03').

    Returns:
        Concatenated pandas DataFrame from all loaded partitions.

    Example:
        df = load_partitioned_parquet(
            'my-bucket', 'telemetry/',
            columns=['device_id', 'sensor_name', 'value'],
            partition_filter='2026-03'
        )
    """
    import time
    import pandas as pd

    s3 = boto3.client("s3", region_name=_get_bucket_region(bucket))

    # Discover partition prefixes
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
    partition_prefixes = sorted(
        p["Prefix"] for p in resp.get("CommonPrefixes", [])
        if "=" in p["Prefix"].rstrip("/").split("/")[-1]  # Only Hive-style partitions (key=value)
    )

    if partition_filter:
        partition_prefixes = [p for p in partition_prefixes if partition_filter in p]

    if max_partitions is not None:
        partition_prefixes = partition_prefixes[:max_partitions]

    dfs = []
    for part_prefix in partition_prefixes:
        # List ALL parquet files in this partition
        resp2 = s3.list_objects_v2(Bucket=bucket, Prefix=part_prefix)
        files = [
            o["Key"] for o in resp2.get("Contents", [])
            if o["Key"].endswith(".parquet")
        ]
        if not files:
            continue

        # Read ALL files in this partition (no data loss)
        for f in files:
            pf = open_parquet_file(f"{bucket}/{f}")
            n_row_groups = pf.metadata.num_row_groups
            for rg in range(n_row_groups):
                table = pf.read_row_group(rg, columns=columns)
                if max_rows_per_partition is not None:
                    table = table.slice(0, max_rows_per_partition)
                dfs.append(table.to_pandas())
            time.sleep(0.1)  # light throttle between files

    if not dfs:
        return pd.DataFrame(columns=columns or [])

    return pd.concat(dfs, ignore_index=True)


def load_all_flat_parquet(
    bucket: str,
    prefix: str,
    columns: list[str] | None = None,
) -> "pd.DataFrame":
    """Load all parquet files under a non-partitioned S3 prefix and concatenate them.

    Use this for flat tables that are split into multiple part files (e.g.
    device_health/, device_master/) but are NOT Hive-partitioned.

    Args:
        bucket: S3 bucket name.
        prefix: S3 prefix (e.g. 'device_health/').
        columns: Optional list of columns to read. None = all columns.

    Returns:
        Concatenated pandas DataFrame from all parquet files found.

    Example:
        health = load_all_flat_parquet('my-bucket', 'device_health/')
        ucm = load_all_flat_parquet('my-bucket', 'device_master/', columns=['device_id', 'device_family'])
    """
    import time
    import pandas as pd

    s3 = boto3.client("s3", region_name=_get_bucket_region(bucket))
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    files = [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".parquet")]

    dfs = []
    for f in files:
        pf = open_parquet_file(f"{bucket}/{f}")
        table = pf.read(columns=columns)
        dfs.append(table.to_pandas())
        time.sleep(0.3)  # avoid S3 throttling

    if not dfs:
        return pd.DataFrame(columns=columns or [])

    return pd.concat(dfs, ignore_index=True)
