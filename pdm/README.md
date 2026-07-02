# pdm — Predictive Maintenance Library

Python library for end-to-end predictive maintenance: data exploration, anomaly detection, and supervised fault prediction from IoT sensor data stored in S3.

## Structure

```
pdm/
├── __init__.py              # Public API: PDMModel, AnomalyDetector, FailureClassifier,
│                            #   RULPredictor, SurvivalPredictor, DatasetMeta
├── base.py                  # PDMModel ABC, TrainResult, PredictionResult
├── training_time.py         # Retraining time estimation utilities
├── example_inference.py     # Reusable single-day inference script (Phase 7)
├── data/                    # Data loading & exploration
│   ├── data_exploration.py  # S3/parquet discovery & schema inspection
│   ├── utils.py             # Feature engineering & caching utilities
│   ├── dataset_schema.py    # DatasetMeta dataclass
│   ├── feature_extraction.py # tsfresh wrapper with manual fallback
│   └── interactions.py      # Auto-generate pairwise feature interactions
├── benchmarks/              # Benchmark suite
│   ├── loaders.py           # C-MAPSS, AI4I, NASA Battery, FEMTO, XJTU-SY, N-CMAPSS
│   ├── benchmark.py         # Benchmark runner CLI
│   ├── download.py          # Dataset download CLI
│   └── baselines.json       # Locked baseline metrics
├── anomaly_detection/       # Unsupervised anomaly detection (Isolation Forest)
│   ├── model.py             # AnomalyDetector(PDMModel)
│   ├── train_anomaly.py     # Train on normal-only data
│   ├── evaluate_anomaly.py  # Evaluate with metrics, plots, synthetic injection
│   ├── synthetic_anomalies.py # Inject synthetic anomalies for validation
│   └── inference_anomaly.py # Score new data & explain anomalies
├── fault_prediction/        # Supervised fault prediction (AutoGluon)
│   ├── model.py             # FailureClassifier(PDMModel)
│   ├── validate_dataset.py  # Dataset validation gate
│   ├── train.py             # AutoML training (binary, multi-label, RUL, survival)
│   └── inference.py         # Predict on new data
├── rul/                     # Remaining Useful Life regression
│   ├── model.py             # RULPredictor(PDMModel) — sliding window + AutoGluon
│   └── train.py             # CLI
├── survival/                # Survival analysis
│   ├── model.py             # SurvivalPredictor(PDMModel) — Cox PH, Weibull AFT, RSF
│   └── train.py             # CLI
├── deployment/              # Batch & real-time serving
│   ├── batch.py             # load_telemetry_window, aggregate_telemetry, predict_proba
│   └── serve.py             # Unified SageMaker handler (all formulations)
└── remote/                  # SageMaker Training Job submission
    ├── submit.py            # Single training job submission
    ├── train_remote.py      # Container entry script (RUL/classification/survival)
    ├── train_multilabel.py  # Container entry script (multi-label)
    └── parallel.py          # Dispatch N experiments, monitor progress, compare
```

## Requirements

Python 3.11+, uv, boto3, pyarrow, pandas, numpy, scikit-learn, autogluon.tabular

---

## pdm.data — Data Loading & Exploration

### data_exploration.py

S3/parquet utilities for discovering and inspecting data.

| Function | Description |
|----------|-------------|
| `explore_bucket(bucket, extension)` | File tree of the bucket |
| `explore_schema(s3_uri, min_coverage, max_columns)` | Schema: dtypes, nulls, cardinality, top values. Accepts optional `prefix=` for flat multi-file tables. |
| `explore_table_summary(bucket, prefix)` | Total row count + file count (metadata only) |
| `discover_join_keys(bucket, prefixes, partitioned, max_partitions)` | Auto-detect shared columns across tables with overlap stats, dtypes, and type_mismatch flags |
| `discover_cross_name_joins(bucket, prefixes, partitioned, max_partitions, min_overlap_pct)` | Find join candidates where column names differ but values overlap |
| `list_eav_attributes(bucket, prefix, attribute_col, group_col, partitioned, max_partitions)` | List distinct attribute values from EAV tables with minimal I/O |
| `validate_join(bucket, left_prefix, right_prefix, left_key, right_key, left_partitioned, right_partitioned, max_partitions)` | Key overlap stats between two tables |
| `load_partitioned_parquet(bucket, prefix, columns, max_partitions, partition_filter)` | Load Hive-partitioned parquet (reads all files per partition, skips `_delta_log/`) |
| `load_all_flat_parquet(bucket, prefix, columns)` | Load all parquet files under a flat prefix |
| `open_parquet_file(s3_uri)` | Returns a PyArrow ParquetFile for custom reads |

**Usage:**

```python
from pdm.data.data_exploration import explore_bucket, explore_schema, discover_join_keys

tree = explore_bucket("s3://my-bucket", extension=".parquet")
schema = explore_schema("s3://my-bucket/telemetry/part-00000.parquet")
candidates = discover_join_keys("my-bucket", prefixes=["telemetry/", "device_master/"], partitioned=["telemetry/"])
```

### utils.py

Data loading, EAV aggregation, feature engineering, and dataset management utilities.

| Function | Description |
|----------|-------------|
| `load_or_cache(name, loader)` | Cache expensive S3 loads as local parquet |
| `load_eav_chunked(...)` | Memory-safe global EAV aggregation (one row per entity). Use for non-temporal or single-snapshot models only. |
| `load_eav_temporal(...)` | Per-(entity, observation_date) EAV aggregation with rolling lookback window. Use for time-series PdM. |
| `pivot_eav(df, entity, attr, value)` | Pivot raw EAV to wide format with aggregations |
| `pivot_precomputed_eav(df, entity, attr)` | Pivot pre-aggregated stats with coverage filter (global) |
| `pivot_precomputed_eav_temporal(df, entity, attr)` | Pivot pre-aggregated temporal stats (per entity × observation_date) |
| `safe_age_days(dates)` | Compute days since dates (handles pandas edge cases) |
| `encode_categoricals(df, cols)` | Label-encode categorical columns |
| `booleans_to_int(df, cols)` | Convert boolean columns (with None) to 0/1 safely |
| `deduplicate_on(df, key)` | Deduplicate DataFrames before joins |
| `sanitize_label_name(title)` | Convert issue title → `label_*` column name |
| `build_multilabel_matrix(...)` | Forward-looking label matrix from health data |
| `temporal_split(df, date_col)` | Temporal train/test split (date-based or unit-based) |
| `drop_zero_variance(df)` | Remove constant-value features |
| `save_dataset(df, output_dir)` | Split + clean + save train.csv, test.csv, dataset.csv |
| `align_to_model(df, metadata_path)` | Reindex columns to match a trained model's feature order |

**Usage:**

```python
from pdm.data.utils import load_or_cache, load_eav_temporal, pivot_precomputed_eav_temporal, temporal_split

telemetry = load_or_cache("telemetry", lambda: load_from_s3(...))
temporal_agg = load_eav_temporal(bucket, "telemetry/", entity_col="device_id", ...)
features = pivot_precomputed_eav_temporal(temporal_agg, entity_col="device_id", attribute_col="sensor_name")
train_df, test_df = temporal_split(features, date_col="observation_date")
```

---

## pdm.anomaly_detection — Unsupervised Anomaly Detection

Trains an Isolation Forest on normal-only data to produce an anomaly scoring model. Detects deviations from normal operating conditions without requiring failure labels.

### train_anomaly.py

Trains an Isolation Forest model on the training set (assumed mostly normal), computes anomaly scores, selects a threshold, and saves all artefacts.

```bash
uv run python pdm/anomaly_detection/train_anomaly.py \
    --train ./data/raw_train.csv \
    --test ./data/raw_test.csv \
    --output ./anomaly_detection/model \
    --contamination 0.05
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--train` | required | Path to training CSV |
| `--test` | required | Path to test CSV |
| `--output` | `./anomaly_detection/model` | Output directory |
| `--contamination` | `0.05` | Expected fraction of anomalies in training data |

**Output artefacts:**

| File | Description |
|------|-------------|
| `isolation_forest.joblib` | Trained Isolation Forest model |
| `scaler.joblib` | StandardScaler fitted on normal data |
| `threshold.json` | Threshold value + strategy |
| `metadata.json` | Feature names, config, training stats |
| `baseline_stats.json` | Per-feature distributions (for drift monitoring) |
| `anomaly_scores_test.csv` | Per-sample scores + binary flag |
| `score_distribution.png` | Histogram with threshold line |
| `metrics.json` | Evaluation metrics + quality gate result |

### evaluate_anomaly.py

Evaluates the anomaly detection model: computes metrics, plots score distributions, and runs a synthetic anomaly injection test.

```bash
uv run python pdm/anomaly_detection/evaluate_anomaly.py \
    --model-dir ./anomaly_detection/model \
    --test ./data/raw_test.csv
```

**Evaluation includes:**
- Score distribution visualization with threshold line
- Supervised metrics (precision, recall, F1, AUROC) when labels are available
- Synthetic injection test: injects spike, drift, and level-shift anomalies and measures detection rate

### synthetic_anomalies.py

Utility for injecting synthetic anomalies into test data for unsupervised validation.

**Anomaly types:**
- **spike** — sudden value jump (point anomaly)
- **drift** — gradual increase over consecutive samples (collective anomaly)
- **level_shift** — permanent offset change (contextual anomaly)

```python
from pdm.anomaly_detection.synthetic_anomalies import inject_anomalies

augmented_df, injection_labels = inject_anomalies(test_df, feature_cols, fraction=0.1)
```

### inference_anomaly.py

Scores new data and identifies top contributing features per anomaly.

```bash
uv run python pdm/anomaly_detection/inference_anomaly.py -n 5
uv run python pdm/anomaly_detection/inference_anomaly.py --input new_data.csv --model-dir ./anomaly_detection/model
uv run python pdm/anomaly_detection/inference_anomaly.py --input new_data.csv --top-features 5 --explain
```

| Argument | Default | Description |
|----------|---------|-------------|
| `-n` | `5` | Number of samples to score |
| `--input` | auto-discover test set | Input CSV file |
| `--model-dir` | `./anomaly_detection/model` | Model directory |
| `--top-features` | `10` | Top contributing features to show |
| `--explain` | off | Show per-feature contribution analysis |

---

## pdm.fault_prediction — Supervised Fault Prediction

AutoGluon-based supervised models for predicting specific failure modes. Auto-detects the formulation from dataset columns.

### Supported Formulations

| Formulation | Detection | Target Columns |
|-------------|-----------|----------------|
| Multi-label classification | `label_*` columns | Multiple binary label columns prefixed with `label_` |
| RUL regression | `RUL` column | Remaining Useful Life as continuous target |
| Binary classification | `machine_failure` column | Single binary target |
| Survival analysis | `duration` + `event` columns | Time-to-event with censoring |

### validate_dataset.py

Dataset validation gate — ensures the dataset is correctly formatted before training.

```bash
uv run python pdm/fault_prediction/validate_dataset.py --data ./experiments/00_baseline/data/train.csv
```

Validates: column types, label format, NaN handling, feature variance, and formulation-specific requirements.

### train.py

Trains an AutoGluon TabularPredictor. Auto-detects the formulation and trains accordingly.

```bash
uv run python pdm/fault_prediction/train.py \
    --train ./experiments/00_baseline/data/train.csv \
    --test ./experiments/00_baseline/data/test.csv \
    --output ./experiments/00_baseline/model \
    --time-limit 120 \
    --presets best
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--train` | required | Path to training CSV |
| `--test` | required | Path to test CSV |
| `--output` | `./model` | Output directory |
| `--time-limit` | `120` | Training time budget in seconds |
| `--presets` | `best` | AutoGluon presets (`best`, `medium_quality`, etc.) |

**Output artefacts:**
- `ag_model/` — AutoGluon model directory
- `metadata.json` — Feature names, formulation, target info
- `metrics.json` — Evaluation metrics + feature importance
- `shap_summary.png` — SHAP feature importance plot

### inference.py

Runs inference on new data using a trained model. Wraps AutoGluon with a `PDMModel` class.

```bash
uv run python pdm/fault_prediction/inference.py --model-path ./experiments/00_baseline/model -n 5
uv run python pdm/fault_prediction/inference.py --input new_data.csv --model-path ./model
```

| Argument | Default | Description |
|----------|---------|-------------|
| `-n` | `5` | Number of samples to predict |
| `--input` | auto-discover test set | Input CSV file |
| `--model-path` | `./model` | Path to model directory |

**Output format** depends on formulation:
- **Classification** — predicted class + probability per label
- **RUL** — predicted remaining useful life value
- **Survival** — predicted duration

---

## Quality Gates

| Model Type | Metric | Threshold |
|------------|--------|-----------|
| Anomaly detection (with labels) | AUROC | ≥ 0.75 |
| Anomaly detection (no labels) | Synthetic injection detection rate | ≥ 80% |
| Binary classification | Test F1 | ≥ 0.50 |
| Multi-label classification | Median test F1 | ≥ 0.50 |
| RUL regression | Test RMSE | < 50% of RUL range |
| Survival analysis | Concordance index | > 0.60 |

---

## Choosing a Formulation

If you manage a fleet of devices (chargers, motors, batteries, pumps, etc.) and want to predict failures, the first question is: **what kind of answer do you need?**

### At a Glance

| | RUL Regression | Failure Classification | Survival Analysis |
|---|---|---|---|
| **Question answered** | "How many days/cycles until this device fails?" | "Will this device fail within the next 7 days?" | "What is the probability this device survives 6 more months?" |
| **Output** | A number (e.g., "42 days left") | Yes/No (or probability per failure mode) | A probability curve over time |
| **Best for** | Scheduling maintenance at the right time | Triggering alerts and work orders | Planning spare parts and workforce across a fleet |
| **Data you need** | Sensor history of devices that ran until failure | Records of which devices failed and which didn't | Same as RUL, but also works when some devices were repaired before failing |

### Decision Flowchart

```
Do you have run-to-failure data (devices that actually failed)?
├── YES → Do you need a precise remaining-life estimate?
│         ├── YES → Use RUL Regression
│         └── NO, just alerts → Use Failure Classification
└── NO (devices are maintained before failure)
    └── Use Survival Analysis
```

Multiple failure modes? Use **multi-label classification** — the library trains one model per label and returns probabilities for each failure type.

### Practical Example: Fleet of EV Chargers

| Scenario | Formulation | Why |
|----------|-------------|-----|
| 50 chargers that failed in the field with full sensor logs | RUL Regression | Learn the degradation pattern, predict remaining life |
| Maintenance logs saying "charger X failed on date Y" but limited sensor data | Failure Classification | Predict which chargers are likely to fail next month |
| Proactively replace chargers at 80% health — none actually "fail" | Survival Analysis | Handles censored data — "survived at least this long" |
| Multiple distinct failure types (overheating, connector wear, board failure) | Multi-label Classification | Independent probability per failure mode |

---

## Preparing Your Own Dataset

The library auto-detects which formulation to use based on your dataset's column structure. Format your data as follows:

### RUL Regression

One row per time step per device, with a continuous `RUL` target:

```csv
unit_id,cycle,sensor_1,sensor_2,sensor_3,RUL
1,1,518.67,642.15,1589.70,125
1,2,518.67,642.35,1591.82,124
1,192,518.67,643.02,1580.03,0
```

- `unit_id` — identifies each device
- `RUL` — remaining cycles/hours/days until failure
- Tip: cap RUL at a maximum (e.g., 125) — early cycles carry no degradation signal

### Binary / Multi-label Classification

One row per observation with binary target(s):

```csv
air_temp,process_temp,rot_speed,torque,tool_wear,machine_failure
298.1,308.6,1551,42.8,0,0
300.4,312.1,1270,65.2,215,1
```

- Binary: single `machine_failure` column (0/1)
- Multi-label: multiple `label_*` columns (e.g., `label_overheating`, `label_connector_wear`)

### Survival Analysis

One row per device with `duration` + `event` columns:

```csv
battery_id,initial_capacity,fade_rate,temp_avg,duration,event
B0005,2.07,-0.006,24.0,168,1
B0007,2.05,-0.003,24.0,120,0
```

- `duration` — observed lifetime
- `event` — 1 if failed, 0 if censored (maintained/retired before failure)

### Auto-Detection Rules

| Column pattern | Detected formulation |
|---|---|
| `label_*` columns | Multi-label classification |
| `RUL` column | RUL regression |
| `machine_failure` column | Binary classification |
| `duration` + `event` columns | Survival analysis |

### Checklist Before Training

1. **No missing target values** — drop or impute NaN in target column(s)
2. **Numeric features** — encode categoricals before input
3. **No data leakage** — don't include future information as features
4. **Reasonable scale** — clip extreme outliers
5. **Enough data** — ≥100 rows for classification/survival, ≥5 units for RUL

---

## Understanding the Metrics

### RUL: RMSE (Root Mean Squared Error)

How far off are predictions from the true remaining life, on average.

- RMSE of 14 = predictions are off by ~14 cycles on average
- Lower is better (0 = perfect)
- Penalizes large errors heavily — one prediction off by 50 hurts more than five off by 10

### Classification: Precision and Recall

**Precision** — "When we predict failure, how often are we right?"
- 0.82 = 82% of failure alerts are real; 18% are false alarms
- High precision = fewer unnecessary maintenance visits

**Recall** — "Of all actual failures, how many did we catch?"
- 0.75 = we catch 75% of failures; 25% are missed
- High recall = fewer surprise breakdowns

The tradeoff: raising one typically lowers the other. For maintenance, recall is usually more important (missed failures are expensive).

### Survival: C-index (Concordance Index)

Pick two devices at random — does the model correctly rank which fails first?

- 0.93 = correct ranking 93% of the time
- 1.0 = perfect, 0.5 = random
- Works even with censored data (devices that haven't failed yet)

### Anomaly Detection: AUROC and Detection Rate

**AUROC** — area under the ROC curve. 1.0 = perfect separation of normal vs anomalous; 0.5 = random.

**Synthetic detection rate** — when we inject known anomalies (spikes, drifts, level shifts), what fraction does the model catch? Used when no failure labels are available.

---

## Explainability

All model types produce interpretability outputs:

- **Fault prediction / RUL**: SHAP feature importance plot (`shap_summary.png`) showing which features drive predictions globally
- **Anomaly detection**: Per-sample top contributing features via `--explain` flag, showing which sensors deviate most from normal

For deployed endpoints, explanations are returned alongside predictions:

```json
{
  "prediction": [1],
  "explanations": [
    {"tool_wear": 2.95, "torque": 1.72, "rot_speed": 1.03}
  ]
}
```

- **Positive values** → push toward failure
- **Negative values** → push toward healthy
- Top contributors per prediction, enabling maintenance teams to understand *which sensor* is driving the alert