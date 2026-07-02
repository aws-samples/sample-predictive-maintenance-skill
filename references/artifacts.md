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
│   └── pdm/                     # Full pdm library (utilities for feature engineering + inference)
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
│   ├── Dockerfile               # Container definition
│   └── requirements.txt         # Pinned inference dependencies
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
- `pdm/` + `runtime.py` are sufficient to run feature engineering in any Python environment with pandas/numpy

## Artifact 3: Inference

**Purpose**: Load model and produce predictions from raw or engineered features.

| File | Contract |
|------|----------|
| `metadata.json` | `{"feature_names": [...], "label_names": [...], "formulation": str, "training_date": str, "n_train_samples": int, "n_features": int}`. Authoritative source for column alignment at inference. |
| `metrics.json` | Per-label metrics (F1, precision, recall, positive_rate), aggregate metrics (mean/median F1). Optional: feature_importance. |
| `baseline_stats.json` | Per-feature: mean, std, q25, q50, q75, min, max. For drift detection at inference time. |
| `ag_model/` | AutoGluon model directory. One subdirectory per label for multi-label. Each contains model weights, version info, and predictor config. |
| `inference.py` | Implements SageMaker hosting contract: `model_fn(model_dir)`, `input_fn(body, content_type)`, `predict_fn(input, model)`, `output_fn(result, content_type)`. Standalone — no imports from `pdm.*`. |
| `serve.py` | Starts Gunicorn with the inference handler. Container entrypoint. |
| `Dockerfile` | Builds the inference container. Pins Python version + AutoGluon version to match training. |
| `requirements.txt` | Exact versions for inference container (autogluon, pandas, numpy, etc.). |

**Contract guarantees**:
- `inference.py` is fully standalone (no `pdm` package dependency in production)
- Input: JSON array of objects or CSV. Columns must be a superset of `metadata.json["feature_names"]` (missing columns filled with 0)
- Output (multi-label): `{"<label>_pred": [0,1,...], "<label>_proba": [0.1, 0.9, ...]}`
- Output (single-label): `{"prediction": [...], "probabilities": [...]}`

**Inference pipeline** (caller responsibility):
```
raw data → engineer_features(raw_df) → align_to_model(df, metadata) → predict
```

## Two Inference Paths

| Path | Feature Engineering | Model Loading | Use Case |
|------|-------------------|---------------|----------|
| **Real-time endpoint** | Caller pre-computes features before sending to endpoint | `inference.py` loads model via `model_fn()` | Low-latency single predictions |
| **Batch inference** | Processing Job runs `runtime.py` + `pdm_utils.py` on raw data | Same `inference.py` pattern inside container | Daily scoring of full fleet |

For **real-time**: the caller (e.g., an application service) imports `runtime.py` and the `pdm` library, calls `engineer_features()`, then sends the engineered row to the SageMaker endpoint.

For **batch**: the SageMaker Processing Job container includes `runtime.py`, `pdm/`, and the model. It reads raw telemetry from S3, engineers features, predicts, and writes results back to S3.

## Artifact 4: Infrastructure

**Purpose**: Deploy the model as a managed AWS service.

| File | Contract |
|------|----------|
| `app.py` | CDK app. Reads context from `cdk.json`. |
| `batch_inference_stack.py` | Creates: EventBridge rule (cron) → Lambda → SageMaker Processing Job. Context vars: `input_bucket`, `ecr_image_uri`, `model_s3_uri`, `instance_type`, `schedule`. |
| `cdk.json` | All deployment configuration. Changing schedule/instance requires only editing this file. |
| `lambda/trigger.py` | Lambda that starts a SageMaker Processing Job with yesterday's date as input. |

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
