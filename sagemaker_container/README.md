# SageMaker Custom Inference Container

Custom container for deploying AutoGluon predictive maintenance models to SageMaker real-time endpoints.

## Why a Custom Container?

| Alternative | Problem |
|---|---|
| AutoGluon DLC (1.3.0) | Uses TorchServe which has OOM issues with multi-label models |
| AutoGluon DLC version mismatch | DLC version ≠ training version → crash on model load |
| sklearn container + pip install | Can't install AutoGluon within SageMaker's 20-minute health check timeout |

This container bakes in the exact AutoGluon version (1.5.0), starts in seconds, and uses a lightweight Flask server.

## Files

| File | Description |
|------|-------------|
| `Dockerfile` | Python 3.12 + AutoGluon 1.5.0 + LightGBM + Flask |
| `serve.py` | Flask server: `/ping` (instant 200) + `/invocations` (lazy model load) |
| `inference.py` | `model_fn` / `input_fn` / `predict_fn` / `output_fn` handlers |
| `requirements.txt` | Reference only — dependencies are baked into the Dockerfile |

## Build & Push

```bash
# Build for Linux (SageMaker requires linux/amd64)
docker build --platform linux/amd64 -t pdm-inference sagemaker_container/

# Tag for ECR
docker tag pdm-inference ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/pdm-inference:latest

# Login to ECR
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com

# Push
docker push ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/pdm-inference:latest
```

## Local Testing

```bash
# Build
docker build --platform linux/amd64 -t pdm-inference sagemaker_container/

# Run (mount your model directory)
docker run -p 8080:8080 -v ./fault_prediction/combined/model:/opt/ml/model pdm-inference

# Health check
curl http://localhost:8080/ping

# Predict (CSV input)
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: text/csv" \
  -d "$(head -2 data/raw_test.csv)"
```

## API

### `GET /ping`
Health check. Returns 200 immediately (no model loading).

### `POST /invocations`
Predict on input data. Lazy-loads model on first call.

**Input formats:**
- `Content-Type: text/csv` — CSV string with header row
- `Content-Type: application/json` — JSON array of records

**Output:** JSON with predictions and probabilities:
```json
{
  "prediction": [0, 1, 0],
  "probability": [0.12, 0.87, 0.23]
}
```

For multi-label models, returns per-label probabilities.

## Version Matching

**CRITICAL**: The Python and AutoGluon versions in the Dockerfile must match what was used during training.

Check your model's versions:
```bash
cat fault_prediction/combined/model/ag_model/*/version.txt
cat fault_prediction/combined/model/environment.json
```

If they differ, update the Dockerfile's `FROM python:X.Y-slim` and `autogluon.tabular==X.Y.Z` lines accordingly.
