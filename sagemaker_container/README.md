# SageMaker Custom Inference Container

Custom container for deploying AutoGluon predictive maintenance models to SageMaker real-time endpoints. The same image runs the daily batch Processing Job.

## Why a Custom Container?

| Alternative | Problem |
|---|---|
| AutoGluon DLC (1.3.0) | Uses TorchServe which has OOM issues with multi-label models |
| AutoGluon DLC version mismatch | DLC version ≠ training version → crash on model load |
| sklearn container + pip install | Can't install AutoGluon within SageMaker's 20-minute health check timeout |

The image installs the **slim pdm library** (`pdm-project[serve]`) at the exact versions in `uv.lock` — the same Python, AutoGluon and scikit-learn the model was trained with — starts in seconds, and uses a lightweight Flask server. Training-only packages (shap, optuna, imbalanced-learn, lifelines, …) are not installed.

Because pdm is installed:

- models whose pickles reference pdm (custom F2/F0.5 or NASA scorers) load;
- columns are aligned with `pdm.data.utils.align_to_model`, like batch inference;
- if the model package contains the project's `runtime.py`, the endpoint accepts **raw rows** and runs `engineer_features` itself.

## Files

| File | Description |
|------|-------------|
| `Dockerfile` | Python 3.12 (`.python-version`) + `pdm-project[serve]` from `uv.lock` + Flask |
| `Dockerfile.dockerignore` | Build-context allowlist (pyproject, lock, `pdm/`, handlers) |
| `serve.py` | Flask server: `/ping` (instant 200) + `/invocations` (lazy model load) |
| `inference.py` | `model_fn` / `input_fn` / `predict_fn` / `output_fn` handlers |

## Build & Push

Build from the **project root** (the directory with `pyproject.toml`, `uv.lock` and `pdm/`). In a skill project, copy this directory in first: `cp -r <SKILL_DIR>/sagemaker_container ./sagemaker_container`.

```bash
# Build for Linux (SageMaker requires linux/amd64)
docker build --platform linux/amd64 -f sagemaker_container/Dockerfile -t pdm-inference .

# Tag for ECR
docker tag pdm-inference ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/pdm-inference:latest

# Login to ECR
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com

# Push
docker push ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/pdm-inference:latest
```

## Local Testing

```bash
# Package the model (add runtime.py to accept raw rows)
mkdir -p /tmp/model_pkg/code
cp -r fault_prediction/combined/model/ag_model fault_prediction/combined/model/metadata.json /tmp/model_pkg/
cp sagemaker_container/inference.py fault_prediction/combined/runtime.py /tmp/model_pkg/code/

docker run -p 8080:8080 -v /tmp/model_pkg:/opt/ml/model pdm-inference

# Health check
curl http://localhost:8080/ping

# Predict on raw rows (runtime.py packaged)
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: text/csv" \
  --data-binary "$(head -2 data/raw_test.csv)"
```

## API

### `GET /ping`
Health check. Returns 200 immediately (no model loading).

### `POST /invocations`
Predict on input data. Lazy-loads model on first call.

**Input formats:**
- `Content-Type: text/csv` — CSV string with header row
- `Content-Type: application/json` — JSON array of records

Rows are **raw** if `code/runtime.py` is packaged, otherwise **engineered features**. Missing feature columns are filled with 0 and logged; extra columns are dropped.

**Output:** JSON with predictions and probabilities:
```json
{
  "prediction": [0, 1, 0],
  "probabilities": [[0.88, 0.12], [0.13, 0.87], [0.77, 0.23]]
}
```

For multi-label models, returns `<label>_pred` and `<label>_proba` (probability of the positive class) per label. Class predictions use each predictor's validation-calibrated decision threshold.

## Version Matching

AutoGluon only loads predictors saved by the same AutoGluon version and Python minor version. The image takes both from the project's `.python-version` and `uv.lock`, so there is nothing to edit here. If a model's versions differ:

```bash
cat fault_prediction/combined/model/environment.json
cat fault_prediction/combined/model/ag_model/*/version.txt
```

the model was trained outside the locked environment — retrain it after `uv sync` instead of changing the container.
