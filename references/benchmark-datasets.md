# Phase 2B: Benchmark Dataset Preparation

## When to Use

Use Path B when the user provides:
- A local folder with a known PdM dataset (C-MAPSS, AI4I, NASA Battery, FEMTO, PHM08, etc.)
- A folder with raw data + documentation (paper PDF, README)
- An explicit formulation mandate ("train a survival model on this data")

**Skip Phases 2 and 3 entirely.** Phase 2B replaces them both.

## Steps

### 2B.1: Understand the Dataset

Read all documentation in the folder:
- README / readme.txt files
- Paper PDFs (extract with pdf2md if needed from the knowledge-acquisition skill)
- Column descriptions, data dictionaries

Determine:
- Raw file format (CSV, TXT space-separated, MAT, Parquet)
- Train/test split convention (if mandated by benchmark)
- Target column semantics
- Evaluation protocol (what metric, what predictions)

### 2B.2: Try Auto-Detection

```python
from pdm.benchmarks.loaders import detect_and_load

meta = detect_and_load(Path("/path/to/dataset/folder"))
if meta:
    print(f"Detected: {meta.name} ({meta.formulation})")
```

If auto-detection succeeds, confirm with the user and proceed to 2B.5.

### 2B.3: Confirm with User

Present:
- Detected formulation and rationale
- Proposed feature engineering approach (sliding window for RUL, raw for classification, etc.)
- Whether to use benchmark's split or custom temporal split
- Any ambiguities from the documentation

**Anomaly Detection**: Build AD only if the data folder documentation explicitly requests it (e.g., mentions anomaly detection, novelty detection, or unsupervised complement). Do NOT add AD by default in Path B.

### 2B.4: Write Adapter (`pdm/raw_dataset.py`)

If auto-detection failed (unknown dataset), generate a script that:
1. Loads raw files from the local folder
2. Applies minimal preprocessing (rename columns, handle missing values)
3. Constructs target column(s) per the benchmark convention
4. Applies the train/test split (benchmark protocol or temporal)
5. Outputs `./data/raw_train.csv`, `./data/raw_test.csv`, `./data/dataset_meta.json`

Use `DatasetMeta` from `pdm.data.dataset_schema`:

```python
from pdm.data.dataset_schema import DatasetMeta

meta = DatasetMeta(
    name="My Dataset",
    source="benchmark",
    formulation="rul",  # or "classification", "multilabel", "survival"
    target_columns=["RUL"],
    feature_columns=feature_cols,
    entity_column="unit_id",
    time_column="cycle",
    split_strategy="per_unit",
    n_train=len(train_df),
    n_test=len(test_df),
    n_features=len(feature_cols),
    evaluation_protocol={"metric": "rmse"},
    anomaly_detection=False,  # True only if explicitly requested in documentation
)
meta.save(Path("./data/dataset_meta.json"))
```

The adapter script MUST be re-runnable and deterministic.

### 2B.5: Validate Output

```bash
uv run python pdm/raw_dataset.py
```

Verify:
- `./data/raw_train.csv` and `./data/raw_test.csv` exist and are non-empty
- Column names match the expected formulation conventions:
  - RUL: must have `RUL`, `unit_id`, `cycle` columns
  - Classification: must have `machine_failure` column
  - Multi-label: must have `label_*` columns
  - Survival: must have `duration` and `event` columns
- `./data/dataset_meta.json` is valid and complete
- Target column distributions look reasonable

```python
from pdm.data.dataset_schema import DatasetMeta
meta = DatasetMeta.load(Path("./data/dataset_meta.json"))
meta.validate()  # Checks columns exist in CSV
```

### 2B.6: Proceed to Phase 4

With the common format in place, proceed directly to Phase 4 (Baselines).

## Column Conventions

| Formulation | Required columns | Notes |
|---|---|---|
| RUL | Features + `RUL` + `unit_id` + `cycle` | One row per (unit × cycle) |
| Classification | Features + `machine_failure` | One row per sample |
| Multi-label | Features + `label_*` columns | One row per sample |
| Survival | Features + `duration` + `event` | One row per unit |
| Anomaly Detection | Features only (labels optional) | One row per sample |

## Known Benchmark Datasets

| Dataset | Formulation | Auto-Detected | Key Files |
|---------|-------------|---------------|-----------|
| C-MAPSS FD001-FD004 | RUL | ✓ | `train_FD0*.txt`, `test_FD0*.txt`, `RUL_FD0*.txt` |
| AI4I 2020 | Classification | ✓ | `ai4i2020.csv` |
| NASA Battery | Survival | ✓ | `_battery_processed.csv`, `*.mat`, or `*Battery*.zip` |
| FEMTO Bearing | RUL | — | Requires custom adapter |
| PHM08 Challenge | RUL | — | Requires custom adapter |

## Benchmarking Lessons Learned

These gotchas were discovered during actual benchmark runs:

### RUL: Window Size and Backend Selection

- **Window=30 is optimal for C-MAPSS FD001**: Our auto-search confirmed window=30 (RMSE 11.96) beats smaller windows (window=15 → RMSE 18.71, window=20 → RMSE 17.43).
- **AutoGluon stacking outperforms Optuna HPO**: On medium-sized tabular data, AutoGluon's `best` presets with stacking (RMSE 11.96) beat Optuna 100-trial ensemble (RMSE 12.96).
- **Always use `--auto-window`** during Phase 5 to confirm the optimal window size rather than assuming.
- Larger windows (50+) hurt performance by diluting the recent degradation trend with older healthy readings.

### Classification: Domain Interaction Features Are High-Impact

- For AI4I 2020, adding `power = torque × rot_speed` and `overstrain = torque × tool_wear` improved F1 from 0.83 → 0.90.
- **Always propose at least one CAAFE-style experiment** that multiplies/divides physically related sensors.
- These features are more effective than oversampling (SMOTE) for tree-based ensembles.

### Survival: Simpler Models Can Win on Small Data

- On the 50-battery dataset, Weibull AFT (baseline) beat Cox PH with extra features (C-index 0.946 vs 0.940).
- Adding derived features to survival models with <50 samples risks overfitting.
- **Quality gate (C-index > 0.65) is usually met by the baseline** — survival experimentation should focus on feature selection rather than feature addition.

### NASA Battery Dataset Distribution

The NASA PCoE Battery dataset is commonly distributed as zip archives. The loader handles:
1. `_battery_processed.csv` cache (fastest — skip extraction)
2. Extracted `*.mat` files (parsed with scipy.io)
3. `*Battery*.zip` files (auto-extracted, then parsed)

Requires `scipy` for `.mat` file parsing. Download the dataset with:
```
uv run python -m pdm.benchmarks.download <base_dir> battery
```

## Phase Detection (for SKILL.md skip logic)

Phase 2B is complete when `./data/dataset_meta.json` exists with `"source": "benchmark"`.
