# Artifact Contracts

Defines the structure, contents, and contracts for all artifacts produced by the PDM skill and saved to `<S3-URI>/<YYYYMMDD_HHMM>/`.

## S3 Layout

```
s3://<output-bucket>/<YYYYMMDD_HHMM>/
├── dataset/                     # 1. Dataset Generation
│   ├── raw_dataset.py           # Reproducible: S3 → raw CSV pipeline
│   ├── raw_train.csv            # Training split
│   └── raw_test.csv             # Test split (temporal holdout)
│
├── training/                    # 2. Training
│   ├── runtime.py               # Feature engineering: raw → model input
│   ├── train_config.json        # Training args (time_limit, presets, formulation)
│   ├── pdm/                     # Full pdm library (utilities for feature engineering + inference)
│   ├── pyproject.toml           # Dependency layout (slim core + extras)
│   ├── uv.lock                  # Exact versions shared by training, container and edge
│   └── .python-version          # Python minor version (AutoGluon requires a match)
│
├── model/                       # 3. Inference — model binaries
│   ├── metadata.json            # Feature names, label names, formulation, training date
│   ├── metrics.json             # Evaluation metrics, feature importance (if computed)
│   ├── baseline_stats.json      # Feature distributions for drift detection
│   └── ag_model/                # AutoGluon model binaries
│       ├── <label_1>/           # One predictor per label (multi-label)
│       │   ├── model.pkl
│       │   └── ...
│       └── <label_N>/
│
├── inference/                   # 3b. Inference runtime (for container deployment)
│   ├── inference.py             # SageMaker model_fn/input_fn/predict_fn/output_fn
│   ├── serve.py                 # Gunicorn entrypoint for container
│   ├── Dockerfile               # Container definition (build from project root)
│   ├── Dockerfile.dockerignore  # Build-context allowlist
│   └── requirements.txt         # Locked pdm-project[serve] requirements (from uv.lock)
│
├── infrastructure/              # 4. Infrastructure (CDK)
│   ├── app.py                   # CDK app entrypoint
│   ├── batch_inference_stack.py # EventBridge → Lambda → SageMaker Processing
│   ├── cdk.json                 # Context variables (bucket, model URI, schedule)
│   ├── requirements.txt         # CDK dependencies
│   └── lambda/
│       └── trigger.py           # Lambda handler that starts the Processing Job
│
└── README.md                    # Documentation: how to use, retrain, deploy
```

## Artifact 1: Dataset Generation

**Purpose**: Reproduce the raw dataset from source data in S3.

| File | Contract |
|------|----------|
| `raw_dataset.py` | Self-contained script. Running `python raw_dataset.py` regenerates `raw_train.csv` and `raw_test.csv` from the source bucket. Must define `BUCKET`, `HORIZON_DAYS`, label configuration, and temporal split logic. Imports from `pdm.data.utils` only. |
| `raw_train.csv` | One row per (entity, observation_date). Columns: entity ID, date, sensor features (EAV-pivoted), metadata features, `label_*` columns. No processing beyond pivot + join. |
| `raw_test.csv` | Same schema as train. Temporal holdout (all dates after the split cutoff). |

**Contract guarantees**:
- `raw_train.csv` and `raw_test.csv` have identical column schemas
- All `label_*` columns are binary (0/1)
- No target leakage (labels are forward-looking from observation date)
- Split is strictly temporal (no entity bleeds across splits)

## Artifact 2: Training

**Purpose**: Transform raw data into model input and configure training.

| File | Contract |
|------|----------|
| `runtime.py` | Defines `engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame`. Pure, deterministic, stateless. Same columns produced for any input row. Handles missing columns gracefully. Used at both training and inference time. |
| `train_config.json` | Records: `{"time_limit": int, "presets": str, "formulation": str, "n_labels": int, "skip_importance": bool}`. Enables exact reproduction of the training run. |
| `pdm/` | Full copy of the pdm library. Provides utilities used by `runtime.py` (`encode_categoricals`, `booleans_to_int`, `align_to_model`, etc.) and data loading functions for batch inference. |

**Contract guarantees**:
- `engineer_features` is idempotent and side-effect-free
- `engineer_features` never drops rows (1-in-1-out)
- Output of `engineer_features` is model-ready when passed through `align_to_model()`
- `pdm/` + `runtime.py` are sufficient to run feature engineering with the slim pdm install (no training-only packages); `uv.lock` + `.python-version` reproduce the exact environment

## Artifact 3: Inference

**Purpose**: Load model and produce predictions from raw or engineered features.

| File | Contract |
|------|----------|
| `metadata.json` | `{"feature_names": [...], "label_names": [...], "formulation": str, "training_date": str, "n_train_samples": int, "n_features": int}`. Authoritative source for column alignment at inference. |
| `metrics.json` | Classification metrics include F1, precision, recall, ROC AUC, average precision, `optimization_priority`, `eval_metric`, the validation-calibrated `decision_threshold`, and `quality_gate` (`passed`, `criterion` = `f1` or `ranking`, `positive_rate`, `reason`); multi-label runs include the same operating-point and gate fields per label plus positive_rate, aggregate mean/median metrics, and an aggregate `quality_gate`. ROC AUC and average precision are `null` when a test split contains only one class. Optional: feature_importance. |
| `baseline_stats.json` | Per-feature: mean, std, q25, q50, q75, min, max. For drift detection at inference time. |
| `ag_model/` | AutoGluon model directory. One subdirectory per label for multi-label. Each contains model weights, version info, and predictor config. |
| `inference.py` | Implements SageMaker hosting contract: `model_fn(model_dir)`, `input_fn(body, content_type)`, `predict_fn(input, model)`, `output_fn(result, content_type)`. Runs on the slim pdm install and may import `pdm.*`. If the model package contains `code/runtime.py`, applies its `engineer_features` to incoming rows. |
| `serve.py` | Starts the Flask server with the inference handler. Container entrypoint. |
| `Dockerfile` | Builds the inference container from the project root: Python from `.python-version`, dependencies of `pdm-project[serve]` from `uv.lock` (hash-checked), then the `pdm/` library. |
| `Dockerfile.dockerignore` | Build-context allowlist. |
| `requirements.txt` | Locked, hashed `pdm-project[serve]` requirements exported from `uv.lock` at save time (record of what the image installs). |

**Contract guarantees**:
- The container installs only the slim inference install (`pdm-project[serve]`) at the training environment's exact versions — no training-only packages
- Input: JSON array of objects or CSV. **Raw rows** (same columns as `raw_test.csv`) when `code/runtime.py` is packaged, otherwise **engineered features**. Columns are aligned with `pdm.data.utils.align_to_model` (missing features filled with 0 and logged, extras dropped)
- Output (multi-label): `{"<label>_pred": [0,1,...], "<label>_proba": [0.1, 0.9, ...]}`
- Output (single-label): `{"prediction": [...], "probabilities": [...]}`

**Inference pipeline** (inside the endpoint when `runtime.py` is packaged; otherwise the caller runs the first step):
```
raw data → engineer_features(raw_df) → align_to_model(df, metadata) → predict
```

## Two Inference Paths

| Path | Feature Engineering | Model Loading | Use Case |
|------|-------------------|---------------|----------|
| **Real-time endpoint** | Endpoint runs the packaged `runtime.py` on raw rows (or caller pre-computes features if `runtime.py` is not packaged) | `inference.py` loads model via `model_fn()` | Low-latency single predictions |
| **Batch inference** | Processing Job runs `runtime.py` with the image's installed pdm library on raw data | Same image as the endpoint | Daily scoring of full fleet |

For **real-time**: package the winning experiment's `runtime.py` with the model (Phase 8.3) and send raw rows; the endpoint uses the same `engineer_features` and pdm version as training, so callers never re-implement feature engineering. `engineer_features` is row-wise (1-in-1-out), so single-row requests produce the same features as training.

For **batch**: the SageMaker Processing Job container includes `runtime.py`, `pdm/`, and the model. It reads raw telemetry from S3, engineers features, predicts, and writes results back to S3.

## Artifact 4: Infrastructure

**Purpose**: Deploy the model as a managed AWS service.

| File | Contract |
|------|----------|
| `app.py` | CDK app. Reads context from `cdk.json`. |
| `batch_inference_stack.py` | Creates: EventBridge rule (cron) → Lambda → SageMaker Processing Job. Context vars: `input_bucket`, `ecr_image_uri`, `model_s3_uri`, `instance_type`, `schedule`. |
| `cdk.json` | All deployment configuration. Changing schedule/instance requires only editing this file. |
| `lambda/trigger.py` | Lambda handler that starts a SageMaker Processing Job with yesterday's date as input. |

**Contract guarantees**:
- All configuration via `cdk.json` context — no hardcoded values in stack code
- `model_s3_uri` points to the `model/` prefix of this artifact bundle
- Stack is idempotent (`cdk deploy` can be run repeatedly)

## What's NOT Saved

| Excluded | Reason |
|----------|--------|
| `.venv/`, `__pycache__/` | Reproducible from `pyproject.toml` |
| `data/_cache_*.parquet` | Intermediate cache, regenerated by `raw_dataset.py` |
| `fault_prediction/experiments/*/` (non-winners) | Discarded experiments add no deployment value |
| `fault_prediction/baseline/` | Superseded by winning experiment |
| `pdm/__pycache__/` | Bytecode cache, regenerated on import |
| `ag_model/*/utils/oof.pkl` | Out-of-fold predictions — training artifact only |
| `log.md`, `data_exploration.md` | Development artifacts, not needed for deployment |
| `anomaly_detection/` | Separate model — saved independently if requested |