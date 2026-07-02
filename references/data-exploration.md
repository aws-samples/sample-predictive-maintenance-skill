# Phase 2: Data Exploration & Strategy

## Key Functions — `pdm/data/data_exploration.py`

| Function | Purpose |
|----------|---------|
| `explore_bucket(bucket, extension)` | File tree of the bucket |
| `explore_schema(s3_uri, min_coverage, max_columns)` | Concise schema: dtypes, nulls, cardinality, truncated top values. Use `min_coverage=0.1` to hide mostly-null columns; `max_columns=40` for wide tables. Accepts optional `prefix=` for flat multi-file tables. |
| `explore_table_summary(bucket, prefix)` | Total row count + file count for flat tables (metadata only, no data loaded) |
| `discover_join_keys(bucket, prefixes, partitioned, max_partitions)` | Auto-detect shared column names across table pairs. Includes `dtypes` and `type_mismatch` flag. |
| `discover_cross_name_joins(bucket, prefixes, partitioned, max_partitions, min_overlap_pct)` | Find join candidates where column NAMES differ but VALUES overlap. |
| `list_eav_attributes(bucket, prefix, attribute_col, group_col, partitioned, max_partitions)` | List distinct attribute values from an EAV table with minimal I/O. |
| `validate_join(bucket, left_prefix, right_prefix, left_key, right_key, left_partitioned, right_partitioned, max_partitions)` | Key overlap stats between two tables |
| `load_partitioned_parquet(bucket, prefix, columns, max_partitions, partition_filter)` | Load Hive-partitioned data (reads all files per partition) |
| `load_all_flat_parquet(bucket, prefix, columns)` | Load all parquet files under a flat prefix |
| `open_parquet_file(s3_uri)` | Low-level: returns a PyArrow ParquetFile for custom reads |

## Step 2.1: Discover Bucket Structure

```python
from pdm.data.data_exploration import explore_bucket, explore_schema, explore_table_summary, validate_join

tree = explore_bucket(bucket, extension='.parquet')
```

Identify unique tables. For partitioned tables, scan ONE representative file. For flat tables, get total counts:
```python
summary = explore_table_summary(bucket, 'device_health/')
```

## Step 2.2: Explore Each Table's Schema

```python
info = explore_schema(f"{bucket}/{path_to_representative_file}")
```

For **flat multi-file tables** (multiple part files, no Hive partitioning): use `explore_table_summary()` first. If total rows >> a single file's rows, use the `prefix=` parameter to sample across files:
```python
summary = explore_table_summary(bucket, 'device_health/')
info = explore_schema(bucket, prefix='device_health/')
```

For **EAV tables**: enumerate distinct attributes cheaply before loading full data:
```python
attrs = list_eav_attributes(bucket, 'telemetry/', attribute_col='sensor_name',
                            group_col='sensor_category', partitioned=True, max_partitions=1)
```

Determine for each table:
- Column names, types, nullability, cardinality
- Whether it's time-series (EAV or wide), device metadata, or label source
- Candidate join keys (shared column names)

For EAV tables: note entity/attribute/value columns, distinct sensors, partition range.

## Step 2.3: Validate Join Keys

**First**, auto-detect candidate join columns across all tables:
```python
from pdm.data.data_exploration import discover_join_keys, discover_cross_name_joins

candidates = discover_join_keys(
    bucket,
    prefixes=['telemetry/', 'device_master/', 'labels/'],
    partitioned=['telemetry/'],
    max_partitions=1,
)

cross = discover_cross_name_joins(
    bucket,
    prefixes=['telemetry/', 'device_master/', 'labels/'],
    partitioned=['telemetry/'],
    max_partitions=1,
    min_overlap_pct=10,
)
```

Then validate the most promising candidates with `validate_join()`:
```python
result = validate_join(
    bucket, left_prefix='telemetry/', right_prefix='device_master/',
    left_key='device_id', right_key='device_id',
    left_partitioned=True, max_partitions=1,
)
```

- If overlap < 10%, the join is broken — look for alternative columns
- Log all validated joins and their overlap % in `log.md`

## Step 2.4: Write `data_exploration.md`

Sections:

1. **Overview of All Tables** — name, file count, row count, purpose
2. **Schema Details Per Table** — columns, types, nullability, key distributions
3. **Candidate Join Keys** — which columns connect tables, type mismatches, overlap %
4. **Table Classification** — time-series, metadata, label source, supplementary
5. **EAV Table Details** (if applicable) — entity/attribute/value columns, sensor list, partition metadata
6. **Data Quality Observations** — nulls, type issues, sparse columns
7. **Feature Candidates** — grouped by signal type (electrical, thermal, behavioral, metadata)
8. **Label Candidates** — all possible targets with source, construction logic, estimated positive rate

### Low-Positives Label Decision Framework

When estimating label positive counts for multi-label/multi-class problems, apply these thresholds:

| Train Positives | Action | Rationale |
|-----------------|--------|-----------|
| ≥ 100 | ✅ Keep | Sufficient for tree-based models to learn patterns |
| 50–99 | ⚠️ Keep with warning | Model may underperform; flag in data_exploration.md |
| 20–49 | 🔀 Merge into related category | Too few for reliable per-class learning; merge with semantically similar label |
| < 20 | ❌ Drop or merge | Cannot learn — will produce random predictions or overfit |

When merging, group by semantic similarity (e.g., related failure modes, similar root causes, or adjacent symptom categories). Document the merge decision and original counts in `data_exploration.md`.

**Ask the user** when a label falls in the merge/drop zone and there's no obvious merge target.

## Step 2.5: Present Questions — STOP AND WAIT

Append `## Open Questions` with structured questions. Format:

```markdown
## Question [N]
[Question text with data context]

A) [Option — rationale/tradeoff]
B) [Option — rationale/tradeoff]
X) Other (please describe after [Answer]: tag below)

[Answer]: 
```

Rules:
- Always include "Other" as last option
- 2–5 meaningful options + Other, mutually exclusive
- Include concrete numbers (positive rate, coverage %, row counts)
- **Do NOT ask questions whose answer can be derived from PdM best practices.** Apply domain conventions directly and document rationale. Examples:
  - Temporal windowing strategy — use rolling windows when data spans < 6 months
  - Train/test split method — always use temporal split for time-series PdM data
  - Feature aggregation granularity — match the label's temporal resolution

**Required topics** (always ask, never skip or infer):
- **Target label** — what should the model predict? (May be answered in user's initial prompt — confirm.)
- **Formulation** — binary / multi-label / RUL / survival? **Recommend** based on data:
  - Run-to-failure with known failure times → **RUL regression**
  - Maintenance/censoring present → **survival analysis**
  - Multiple independent failure modes labeled → **multi-label classification**
  - Binary health/failure flags → **binary classification**
- **Feature scope** — which data sources should be used as model input features? List ALL available tables and ask the user to confirm inclusion. Do NOT assume a table is excluded just because the user didn't mention it — they may not know what's available.
- **Prediction window (horizon)** — how many days ahead should the model predict failures? This is an operational decision driven by maintenance lead times and business processes, NOT by data properties. Always ask explicitly. Propose a default (e.g., 7 or 14 days) with rationale, but require confirmation.
- **Anomaly Detection** — also build an unsupervised Isolation Forest model to catch novel failure modes? **Recommend yes** when the data has high-dimensional sensor readings or when the labelled failure modes are unlikely to be exhaustive.
- **Optimization priority** — does the user care more about recall (catching all failures), precision (minimizing false alarms), or balanced F1? This is an operational decision driven by maintenance capacity, safety criticality, and false-alarm tolerance. It affects the training eval metric and inference threshold selection.

**When the answer is obvious from context, propose it instead of asking open-ended — but still STOP AND WAIT for confirmation.** Use a confirmation format:

```markdown
## Question [N]
Based on [data evidence], I recommend [choice] because [reasoning].

A) **Confirm recommendation** — proceed with [choice]
B) [Alternative option with tradeoff]
X) Other (please describe)

[Answer]:
```

Examples of when to propose directly (but still wait for confirmation):
- User says "use X as labels" → formulation is implied (multi-label if multiple label values, classification if binary)
- High-dimensional sensor data exists → recommend anomaly detection = yes
- Only telemetry + metadata tables available → feature scope = telemetry + metadata (but list all tables and confirm)
- Multiple independent failure types labeled → multi-label classification

**Even when proposing, you MUST still stop and wait.** Never proceed to Phase 3 without the user's explicit go-ahead.

Only ask genuinely open-ended questions when the data reveals ambiguities the user must resolve (e.g., prediction horizon, which failure modes are actionable).

**Additional questions** — ask when data reveals genuine ambiguities:
- Which failure modes are actionable vs. cosmetic?
- Priority: recall (catch all failures) vs precision (minimize false alarms)?
- How to handle rare failure modes?

Ask **all questions that the data cannot answer on its own** — typically 4–6 total.

After appending, tell the user and **STOP**.

**How the user can answer:**
- **In chat** — the user replies with letter choices or descriptions. Map their answers to the `[Answer]:` tags in `data_exploration.md` and fill them in.
- **By editing the file** — the user edits `data_exploration.md` directly to fill in `[Answer]:` tags. Re-read the file when they say they're done.

Either path is valid. Do NOT require one specific method.

## Step 2.6: Validate Answers

Once the user confirms:
1. Read `[Answer]:` tags — if any empty, ask to fill in
2. Check for contradictions (e.g. RUL chosen but label has no temporal component)
3. If contradictions found, append `## Clarification Questions` and ask

## Step 2.7: Record Decisions

Append `## User Decisions` to `data_exploration.md`:

```markdown
## User Decisions

- **Target label**: [chosen label, source table, construction logic]
- **Formulation**: [classification/regression/survival]
- **Feature scope**: [which tables]
- **Anomaly detection**: [yes/no]
- **Temporal alignment**: [if asked]
- **Additional notes**: [free-text from "Other" answers]
```

**Only then proceed to Phase 3.**

## Error Recovery

| Failure | Recovery |
|---------|----------|
| S3 access denied | Check credentials, bucket name, region |
