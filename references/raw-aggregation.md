# Raw Data Aggregation

Load raw data from S3, join tables, construct labels, and produce a **raw** train/test split. This phase performs NO feature engineering — it outputs the rawest usable form of the data that all downstream experiments will consume.

## ⚠️ Gotchas

- **No data loss**: Never discard data. Load ALL files per partition, ALL partitions, ALL sensors. Use `load_eav_chunked()` for memory-constrained environments — never skip files or subsample rows.
- **Join key type mismatches**: Device IDs are often `int64` in one table and `string` in another. Always cast both sides to string before joining.
- **Safe age computation**: Never use `(pd.Timestamp.now() - col).dt.days` — fails on `TimedeltaIndex`. Use: `pd.to_datetime(pd.Series(dates), errors='coerce')` then `.dt.days`.
- **Boolean columns with None**: Never use `.astype(int)` on boolean columns from parquet — they may contain `None` which raises `TypeError`. Use `.map({True: 1, False: 0, None: 0}).fillna(0).astype(int)`.
- **Metadata deduplication**: Always deduplicate metadata/master tables before joining — duplicates inflate row counts silently.
- **Multi-label label naming**: All label columns MUST be prefixed with `label_` for auto-detection by `validate_dataset.py` and `train.py`.
- **`partition_filter` is a substring**: `load_partitioned_parquet(partition_filter=...)` accepts a string for substring matching (e.g. `"2026-04"`) — NOT a callable. To load a date range, call once per day in a loop.

## Key Functions — `pdm/data/utils.py`

| Function | Purpose |
|----------|---------|
| `load_or_cache(name, loader)` | Cache expensive S3 loads as parquet |
| `load_eav_chunked(...)` | Memory-safe EAV aggregation — **global** (one row per entity). Use ONLY for non-temporal data. |
| `load_eav_temporal(...)` | **Per-observation-date** EAV aggregation with rolling lookback window. Use for time-series PdM. |
| `pivot_eav(df, entity, attr, value)` | Pivot raw EAV to wide format with aggregations |
| `pivot_precomputed_eav(df, entity, attr)` | Pivot pre-aggregated stats (global, one row per entity) |
| `pivot_precomputed_eav_temporal(df, entity, attr)` | Pivot pre-aggregated temporal stats (one row per entity × observation_date) |
| `safe_age_days(dates)` | Compute days since dates (handles all pandas edge cases) |
| `encode_categoricals(df, cols)` | Label-encode categoricals |
| `booleans_to_int(df, cols)` | Convert booleans (with None) to 0/1 |
| `deduplicate_on(df, key)` | Deduplicate before joins |
| `load_and_prepare_metadata(...)` | Load flat table → dedup → cast ID → booleans → age features |
| `sanitize_label_name(title)` | Convert issue title → `label_*` column name |
| `build_multilabel_matrix(...)` | Forward-looking label matrix from health data |
| `temporal_split(df, date_col)` | Temporal train/test split (date-based or unit-based) |
| `drop_zero_variance(df)` | Remove constant features |
| `save_dataset(df, output_dir)` | Split + clean + save train.csv, test.csv, dataset.csv |
| `align_to_model(df, metadata_path)` | Reindex columns to match trained model's feature order |

## Goal

Produce `./data/raw_train.csv` and `./data/raw_test.csv` containing:
- One row per observation unit (device × time window, or unit × cycle)
- All raw signal values (pivoted from EAV if needed, but NOT transformed)
- The target/label column(s) as defined in User Decisions
- A temporal train/test split

These raw files are the **single source of truth** for all experiments. Feature engineering happens per-experiment in Phase 5.

## Files to Write

### `./pdm/raw_dataset.py` — Raw dataset generation script

Replaces the old `dataset.py`. Responsible for:
1. Loading all source tables from S3 (using `load_or_cache`)
2. Joining tables on validated keys (from Phase 2)
3. Constructing the target/label column(s) per User Decisions
4. Pivoting EAV data to wide format (raw stats only: mean, std, min, max)
5. Performing temporal train/test split
6. Saving raw outputs

**Does NOT do**: interaction features, derived ratios, domain-specific transforms, encoding, or any feature selection.

## Data Loading Patterns

Use all bundled utilities:

```python
from pdm.data.utils import load_or_cache, load_eav_temporal, pivot_precomputed_eav_temporal
from pdm.data.utils import load_eav_chunked, pivot_precomputed_eav  # only for non-temporal use
from pdm.data.utils import deduplicate_on, booleans_to_int, safe_age_days
from pdm.data.utils import build_multilabel_matrix, temporal_split
from pdm.data.data_exploration import (
    load_partitioned_parquet, load_all_flat_parquet, open_parquet_file, _get_bucket_region
)
```

**Core principle: NEVER discard data.** Load ALL files, ALL partitions, ALL sensors.

> **Note**: All examples below use `device_id` as a placeholder entity column. Substitute with the actual column name discovered in Phase 2 (e.g. `charger_id`, `machine_id`, `asset_id`).

### Choosing Between `load_eav_temporal` and `load_eav_chunked`

| Criterion | `load_eav_temporal` | `load_eav_chunked(partition_col=...)` |
|-----------|--------------------|-----------------------------------------|
| Use when | Partitions are small (<5M rows each) | Partitions are large (>5M rows each) |
| How it works | Loads each partition multiple times (rolling window) | Loads each partition once, returns per-partition stats |
| Output | Per (entity, observation_date) with lookback window | Per (entity, partition_date) — one stat row per day |
| Memory | O(lookback_days × entities × sensors) | O(1 partition) at a time |
| Speed | Slower — rereads partitions for overlapping windows | Faster — single pass |

**Heuristic**: Check one partition's row count via `explore_table_summary()` or `explore_schema()`. If a single partition exceeds **5M rows**, prefer `load_eav_chunked(partition_col="_observation_date")` followed by `pivot_precomputed_eav_temporal()`. The features will be per-partition-date stats rather than rolling-window stats, but for daily partitions this is usually equivalent.

```python
# FAST PATH: Large partitions (>5M rows each) — single pass, per-day stats
telemetry_daily = load_or_cache("telemetry_daily", lambda: load_eav_chunked(
    bucket, 'telemetry/',
    entity_col='device_id', attribute_col='sensor_name', value_col='sample_value',
    open_parquet_file_fn=open_parquet_file, get_bucket_region_fn=_get_bucket_region,
    partition_col="_observation_date",  # preserves per-date granularity
))
raw_signals = load_or_cache("raw_signals_daily", lambda: pivot_precomputed_eav_temporal(
    telemetry_daily, entity_col='device_id', attribute_col='sensor_name',
    date_col='_observation_date', stat_cols=('mean', 'std', 'max', 'min'), min_coverage=0.05,
))
```

### EAV Tables — Temporal (DEFAULT for time-series PdM)

When the model has **per-date observations** (e.g., multi-label with one label row per device × date), telemetry MUST be aggregated per (device, observation_date) with a lookback window:

```python
# Compute observation_dates first (from label construction, see below)
telemetry_temporal = load_or_cache("telemetry_temporal", lambda: load_eav_temporal(
    bucket, 'telemetry/',
    entity_col='device_id', attribute_col='sensor_name', value_col='sample_value',
    observation_dates=observation_dates, lookback_days=7,
    open_parquet_file_fn=open_parquet_file, get_bucket_region_fn=_get_bucket_region,
))

raw_signals = load_or_cache("raw_signals_temporal", lambda: pivot_precomputed_eav_temporal(
    telemetry_temporal, entity_col='device_id', attribute_col='sensor_name',
    stat_cols=('mean', 'std', 'max', 'min'), min_coverage=0.05,
))
# raw_signals has columns: [device_id, _observation_date, sensor1_mean, sensor1_std, ...]
```

This ensures features VARY over time for the same device — critical for learning "when will this device fail?" rather than just "which devices fail?"

### EAV Tables — Global (ONLY for single-snapshot models)

Use `load_eav_chunked` ONLY when the model has ONE row per device (no temporal observation dimension):

```python
# ⚠️ Do NOT use this when labels have a per-date structure!
telemetry_agg = load_or_cache("telemetry_agg", lambda: load_eav_chunked(
    bucket, 'telemetry/',
    entity_col='device_id', attribute_col='sensor_name', value_col='sample_value',
    open_parquet_file_fn=open_parquet_file, get_bucket_region_fn=_get_bucket_region,
))

raw_signals = load_or_cache("raw_signals", lambda: pivot_precomputed_eav(
    telemetry_agg, entity_col='device_id', attribute_col='sensor_name',
    stat_cols=('mean', 'std', 'max', 'min'), min_coverage=0.05,
))
```

### Metadata / Flat Tables

```python
metadata = load_or_cache("metadata", lambda: load_all_flat_parquet(bucket, 'device_master/'))
metadata = deduplicate_on(metadata, "device_id")
```

## Target Construction

Follow User Decisions exactly. For multi-label:

```python
from pdm.data.utils import build_multilabel_matrix

label_matrix = build_multilabel_matrix(
    devices=device_list, observation_dates=observation_dates,
    events_df=events_df, device_col="device_id",
    date_col="detection_date", issue_col="issue_title",
    kept_issues=kept_issues, horizon_days=7,
)
```

## Joining and Population Filtering

```python
# TEMPORAL JOIN: merge on BOTH device and observation date
# raw_signals from pivot_precomputed_eav_temporal has [device_id, _observation_date, ...]
dataset = label_matrix.merge(raw_signals, on=["device_id", "_observation_date"], how="inner")
# LEFT join for supplementary sources — metadata is static (one row per device)
dataset = dataset.merge(metadata, on="device_id", how="left")
```

**⚠️ If using the global (non-temporal) aggregation**: the join is only on `device_id` — this duplicates the same features across all observation dates for each device. This is WRONG for time-series PdM.

## Raw Output — Allowed vs Forbidden Transforms

**Allowed:**
- Cast join keys to string for consistency
- Convert booleans to 0/1 (via `booleans_to_int`)
- Basic type coercion (ensure numerics are numeric)
- Drop columns with <5% coverage

**Forbidden:**
- Interaction features (e.g., `temp_A - temp_B`)
- Ratios or derived metrics
- Encoding categoricals (keep as-is for experiments to handle)
- Feature selection based on importance
- Any domain-specific transforms

## Temporal Split and Output

**⚠️ Label leakage at split boundary:** When labels use a forward-looking horizon (e.g., `horizon_days=14`), the last `horizon_days` of training data can have labels that peek into the test period. Use `temporal_split(..., horizon_days=N)` to enforce a gap between the last train date and first test date. Without this gap, the model trains on labels that were constructed using information from the test period.

```python
from pdm.data.utils import temporal_split
import os

# horizon_days ensures no label look-ahead leakage across the split boundary
train, test = temporal_split(dataset, date_col="_observation_date", train_frac=0.8,
                             horizon_days=HORIZON_DAYS)

os.makedirs("./data", exist_ok=True)
train.to_csv("./data/raw_train.csv", index=False)
test.to_csv("./data/raw_test.csv", index=False)
```

Output:
- `./data/raw_train.csv` — raw training data (~80% earliest)
- `./data/raw_test.csv` — raw test data (~20% latest)

## Running

```bash
uv run python pdm/raw_dataset.py
```

## Validation

At the end of `raw_dataset.py`:

```python
assert not train.empty, "Empty training set"
assert not test.empty, "Empty test set"
assert train.columns.tolist() == test.columns.tolist(), "Column mismatch"

label_cols = [c for c in train.columns if c.startswith("label_")]
for lc in label_cols:
    print(f"  {lc}: pos_rate={train[lc].mean():.3%} (n_pos={int(train[lc].sum())})")
```

Do NOT run `validate_dataset.py` here — that validates processed features, not raw data.

**After raw aggregation completes, update `log.md`** with: raw dataset shape, number of columns, label positive rates, split cutoff date, train/test sizes.

**Then proceed to Phase 4 (Baselines).**

## Error Recovery

| Failure | Recovery |
|---------|----------|
| OOM / S3 throttle | Re-run `raw_dataset.py` (caches reused), use `load_eav_chunked()` |
