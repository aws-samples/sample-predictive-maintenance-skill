# Phase 8: Deploy to SageMaker

**Input bucket is read-only.** All deployment artifacts (model, batch code, predictions) go to the **output bucket**. Never write to the input bucket.

**Read [references/deployment.md](references/deployment.md) for full guidelines** — covers endpoint deployment, batch inference, and infrastructure.

After saving to S3, **ask the user which deployment mode they need:**

> How will you consume predictions from this model?
>
> A) **Real-time endpoint** — low-latency, prediction-by-prediction via API call (persistent infrastructure)
> B) **Batch inference** — daily scheduled job processing yesterday's telemetry data from S3 (no persistent infrastructure)
> C) **Both** — real-time endpoint + daily batch job

Then execute Phase 8A (endpoint), Phase 8B (batch), or both depending on the answer. Region defaults to `$AWS_REGION` environment variable. Do NOT deploy without explicit confirmation.

---

## Phase 8A: Real-Time Endpoint

Deploy the best fault prediction model as a real-time SageMaker endpoint using a custom Docker container.

**Ask the user for:**
1. **SageMaker execution role ARN** — must have S3 read + SageMaker + ECR permissions
2. **Endpoint name** (default: `pdm-<project-name>`)

### Why a Custom Container

Pre-built containers don't work reliably for AutoGluon models:
- **AutoGluon DLC** uses TorchServe (OOM with multi-label models, version mismatch with newer AG)
- **sklearn container** can't install AutoGluon within the health check timeout (~20 min)
- **Custom container** bakes in the exact versions, starts in seconds, works every time

### ⚠️ Gotchas (Endpoint)

- **Python/AutoGluon version mismatch**: Container MUST match the training environment exactly. Check `environment.json` and `version.txt` before building.
- **`pickle` error in `load_child`**: Use the full `ag_model/` directory, never `clone_for_deployment` — child models need their pkl files.
- **Cold start on first invocation**: First call loads all predictors (~10-30s). This is normal — subsequent calls are instant.
- **Container deps**: Dockerfile must include `boto3`, `pyarrow`, `tqdm`, `libgomp1` in addition to `autogluon`.

### Step 8.1: Verify Version Requirements

Check what versions the model was trained with — the container MUST match:

```bash
cat fault_prediction/baseline/model/environment.json | python3 -c "import json,sys; print(json.load(sys.stdin)['python_version'])"
cat fault_prediction/baseline/model/ag_model/*/version.txt | head -1
```

Update `<SKILL_DIR>/sagemaker_container/Dockerfile` if versions differ from defaults (Python 3.12, AutoGluon 1.5.0).

### Step 8.1b: Pre-check Existing Resources

Before creating infrastructure, check what already exists to avoid errors and redundant work:

```bash
# IAM role
aws iam get-role --role-name SageMakerExecutionRole --query 'Role.Arn' --output text 2>/dev/null && echo "✅ Role exists" || echo "❌ Need to create"

# ECR repository
aws ecr describe-repositories --repository-names pdm-inference --region $REGION 2>/dev/null && echo "✅ ECR exists" || echo "❌ Need to create"

# Existing endpoint (update vs create)
aws sagemaker describe-endpoint --endpoint-name $ENDPOINT_NAME --region $REGION 2>/dev/null && echo "✅ Endpoint exists — will update" || echo "❌ Will create new"
```

Skip creation steps for resources that already exist. For endpoints already `InService`, use `UpdateEndpoint` for zero-downtime deployment.

### Step 8.2: Build and Push Container

```bash
REGION="${AWS_REGION:-eu-central-1}"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REPO_NAME="pdm-inference"
IMAGE="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:latest"

aws ecr create-repository --repository-name $REPO_NAME --region $REGION 2>/dev/null || true
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com

docker build --platform linux/amd64 -t $REPO_NAME <SKILL_DIR>/sagemaker_container/
docker tag $REPO_NAME:latest $IMAGE
docker push $IMAGE
```

### Step 8.3: Package model.tar.gz

```bash
BEST_MODEL_DIR="./fault_prediction/baseline/model"  # or experiments/<best>/model

rm -rf /tmp/model_pkg && mkdir -p /tmp/model_pkg/code
cp -r "$BEST_MODEL_DIR/ag_model" /tmp/model_pkg/
cp "$BEST_MODEL_DIR/metadata.json" /tmp/model_pkg/
cp <SKILL_DIR>/sagemaker_container/inference.py /tmp/model_pkg/code/

cd /tmp/model_pkg && tar czf /tmp/model.tar.gz . && cd -
aws s3 cp /tmp/model.tar.gz s3://<bucket>/endpoint/model.tar.gz
```

### Step 8.4: Test Locally with Docker (recommended)

```bash
IMAGE="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/pdm-inference:latest"

docker run -d --name sm-local-test -v /tmp/model_pkg:/opt/ml/model -p 8080:8080 $IMAGE
sleep 5 && curl http://localhost:8080/ping
curl -X POST http://localhost:8080/invocations -H "Content-Type: application/json" \
  -d '[{"feature_1": 0.0, "feature_2": 1.0}]'
docker rm -f sm-local-test
```

### Step 8.5: Create IAM Role (if needed)

```bash
aws iam create-role --role-name SageMakerExecutionRole \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"sagemaker.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam attach-role-policy --role-name SageMakerExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
aws iam attach-role-policy --role-name SageMakerExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
```

### Step 8.6: Deploy to SageMaker

```python
"""deploy_endpoint.py — Deploy AutoGluon model to SageMaker real-time endpoint."""
import os, time, boto3

REGION = os.environ.get("AWS_REGION", "eu-central-1")
ACCOUNT = boto3.client("sts").get_caller_identity()["Account"]
ROLE_ARN = f"arn:aws:iam::{ACCOUNT}:role/SageMakerExecutionRole"
ENDPOINT_NAME = "<endpoint-name>"
MODEL_DATA_URL = "s3://<bucket>/endpoint/model.tar.gz"
IMAGE = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/pdm-inference:latest"

sm = boto3.client("sagemaker", region_name=REGION)

model_name = f"pdm-{int(time.time())}"
sm.create_model(ModelName=model_name, PrimaryContainer={"Image": IMAGE, "ModelDataUrl": MODEL_DATA_URL}, ExecutionRoleArn=ROLE_ARN)

config_name = f"{model_name}-config"
sm.create_endpoint_config(EndpointConfigName=config_name, ProductionVariants=[{
    "VariantName": "AllTraffic", "ModelName": model_name,
    "InstanceType": "ml.m5.xlarge", "InitialInstanceCount": 1,
    "ContainerStartupHealthCheckTimeoutInSeconds": 300,
}])

try:
    desc = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
    if desc["EndpointStatus"] == "InService":
        sm.update_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
    else:
        sm.delete_endpoint(EndpointName=ENDPOINT_NAME); time.sleep(30)
        sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
except sm.exceptions.ClientError:
    sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)

while True:
    status = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)["EndpointStatus"]
    print(f"  {status}")
    if status in ("InService", "Failed"): break
    time.sleep(30)
```

### Step 8.7: Verify Endpoint

```python
import os, boto3, json
runtime = boto3.client("sagemaker-runtime", region_name=os.environ.get("AWS_REGION", "eu-central-1"))
response = runtime.invoke_endpoint(EndpointName="<endpoint-name>", ContentType="application/json",
    Body=json.dumps([{"feature_1": 0.0, "feature_2": 1.0}]))
print(json.loads(response["Body"].read()))
```

### Endpoint Key Details

- **Instance type**: `ml.m5.xlarge` (16GB RAM); `ml.m5.large` ok for single-label
- **Deployment time**: ~5 minutes
- **Cold start**: First invocation ~10-30s, then instant
- **Zero-downtime update**: Use `UpdateEndpoint` when already `InService`

---

## Phase 8B: Batch Inference

Deploy a daily batch inference job via SageMaker Processing Job.

**Key performance requirement:** Use `predict_proba()` ONLY — never call `predict()` or `feature_importance()` in the batch path.

**Ask the user for:**
1. **SageMaker execution role ARN** (skip if already provided for 8A)
2. **Input data bucket** (default: the input bucket from Phase 2)

Output: `s3://{input-bucket}-predictions/{YYYYMMDD}/predictions.csv`

### ⚠️ Gotchas (Batch Inference)

- **`partition_filter` is a substring, not a callable**: `load_partitioned_parquet(partition_filter=...)` accepts ONLY a string. To load a date range, call once per day in a loop. Use `pdm.deployment.batch.load_telemetry_window`.
- **Proba-only for speed**: Never call `predict()` + `predict_proba()` together — `predict_proba()` alone suffices (derive class via `> 0.5`).
- **`python -u` for progress**: Always run with `python -u` so tqdm progress bars render in real time.
- **Processing Job environment detection**: Script must detect `/opt/ml/processing/model` vs local paths. Use `os.path.isdir("/opt/ml/processing/model")`.
- **Processing Job instance sizing**: `ml.m5.xlarge` (16GB) OOMs with models >500MB + telemetry. Use `ml.m5.2xlarge` (32GB).

### Step 8B.1: The `pdm/deployment/` Module

Reusable batch utilities at `pdm/deployment/batch.py`:

| Function | Purpose |
|----------|---------|
| `load_telemetry_window(bucket, prefix, target_date, lookback_days, region)` | Load telemetry for a date range (tqdm progress) |
| `aggregate_telemetry(telemetry_df, entity_col)` | Per-device aggregation → wide format |
| `predict_proba(model_dir, features_df)` | Multi-label proba-only prediction (tqdm per label) |
| `score_anomalies(model_dir, features_df)` | Isolation Forest scoring → anomaly_score + is_anomaly |
| `explain_anomalies(model_dir, features_df, top_k=5)` | Z-score deviations from baseline_stats.json |

### Step 8B.2: Generate `batch_inference.py` (Fault Prediction)

```python
"""batch_inference.py — Daily batch predictions."""
import os, sys
import boto3, pandas as pd
from pdm.deployment.batch import load_telemetry_window, aggregate_telemetry, predict_proba
from pdm.data.data_exploration import load_all_flat_parquet
from pdm.data.utils import deduplicate_on, safe_age_days, booleans_to_int

INPUT_BUCKET = os.environ.get("INPUT_BUCKET", "<input-bucket>")
REGION = os.environ.get("AWS_REGION", "eu-central-1")
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", f"{INPUT_BUCKET}-predictions")
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "7"))

# Detect Processing Job environment vs local
if os.path.isdir("/opt/ml/processing/model"):
    MODEL_DIR = "/opt/ml/processing/model"
    sys.path.insert(0, "/opt/ml/processing/code")
else:
    MODEL_DIR = "./fault_prediction/baseline/model"
```

Customize: `INPUT_BUCKET`, `<entity_col>`, `load_metadata()`, `LOOKBACK_DAYS`.

### Step 8B.3: Generate `batch_inference_anomaly.py` (Anomaly Detection)

Anomaly detection batch inference follows the same telemetry loading pipeline but uses the Isolation Forest model and includes feature-level explanations.

```python
"""batch_inference_anomaly.py — Daily anomaly detection for devices.

Loads the last day of telemetry from S3, runs Isolation Forest scoring,
and uploads predictions (with feature explanations) to S3.

Usage:
    uv run python batch_inference_anomaly.py
    TARGET_DATE=2026-04-25 uv run python batch_inference_anomaly.py
"""
import json, os, sys
from datetime import date, timedelta

import boto3, pandas as pd
from pdm.deployment.batch import (
    load_telemetry_window, aggregate_telemetry, score_anomalies, explain_anomalies,
)
from pdm.data.data_exploration import load_all_flat_parquet
from pdm.data.utils import deduplicate_on, safe_age_days, booleans_to_int
from pdm.fault_prediction import baseline_engineer_features

INPUT_BUCKET = os.environ.get("INPUT_BUCKET", "<input-bucket>")
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", f"{INPUT_BUCKET}-predictions")
REGION = os.environ.get("AWS_REGION", "eu-central-1")
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "1"))
MODEL_DIR = os.environ.get("AD_MODEL_DIR", "./anomaly_detection/baseline/model")


def load_metadata():
    df = load_all_flat_parquet(INPUT_BUCKET, "<metadata-prefix>/")
    df["<entity_col>"] = df["<entity_col>"].astype(str)
    df = deduplicate_on(df, "<entity_col>")
    # Select and transform metadata columns (booleans, age, etc.)
    # ... (project-specific)
    return df


def main():
    target_date = os.environ.get("TARGET_DATE", str(date.today() - timedelta(days=1)))
    print(f"=== Anomaly Detection Batch Inference: {target_date} ===\n")

    # 1. Load telemetry
    telemetry = load_telemetry_window(INPUT_BUCKET, "<telemetry-prefix>/", target_date, LOOKBACK_DAYS, REGION)

    # 2. Aggregate per device
    features = aggregate_telemetry(telemetry, entity_col="<entity_col>")

    # 3. Join metadata + engineer features
    metadata_df = load_metadata()
    features = features.merge(metadata_df, on="<entity_col>", how="left")
    entity_ids = features["<entity_col>"].copy()
    features_eng = baseline_engineer_features(features, drop_cols=["<entity_col>"])

    # 4. Score + explain (uses pdm.deployment.batch library functions)
    scores_df = score_anomalies(MODEL_DIR, features_eng)
    explanations = explain_anomalies(MODEL_DIR, features_eng)

    # 5. Build and upload predictions
    threshold = json.loads(open(os.path.join(MODEL_DIR, "threshold.json")).read())["threshold"]
    predictions = pd.DataFrame({
        "<entity_col>": entity_ids.values,
        "prediction_date": target_date,
        "anomaly_score": scores_df["anomaly_score"].values,
        "is_anomaly": scores_df["is_anomaly"].values,
        "threshold": threshold,
        "top_anomalous_features": explanations,
    })

    output_key = f"anomaly-detection/{target_date.replace('-', '')}/predictions.csv"
    predictions.to_csv("/tmp/ad_predictions.csv", index=False)
    boto3.client("s3", region_name=REGION).upload_file(
        "/tmp/ad_predictions.csv", OUTPUT_BUCKET, output_key
    )

    n_anomalies = predictions["is_anomaly"].sum()
    print(f"✅ Saved {len(predictions)} scores ({n_anomalies} anomalies) to s3://{OUTPUT_BUCKET}/{output_key}")


if __name__ == "__main__":
    main()
```

**Key differences from fault prediction batch inference:**

| Aspect | Fault Prediction | Anomaly Detection |
|--------|------------------|-------------------|
| Model type | AutoGluon ensemble (`predict_proba`) | Isolation Forest (`score_samples`) |
| Lookback window | 7 days (default) | 1 day (default) |
| Output columns | `{label}_proba` per failure mode | `anomaly_score`, `is_anomaly`, `top_anomalous_features` |
| Explanations | None (batch speed priority) | Z-score deviations from `baseline_stats.json` |
| Output S3 path | `predictions/{YYYYMMDD}/predictions.csv` | `anomaly-detection/{YYYYMMDD}/predictions.csv` |
| Instance size | `ml.m5.2xlarge` (32GB, large ensemble) | `ml.m5.xlarge` (16GB, lightweight model) |

**Anomaly detection explanation column format:**
```
sensor_wifi_memory_min(z=8.1); internal_meter_voltage_std(z=7.7); dbus_memory_mean(z=5.2); ...
```
Each entry is `feature_name(z=N)` where z is the number of standard deviations from the training baseline mean. Higher z → more anomalous for that feature.

### Step 8B.4: Test Locally

```bash
uv run python -u batch_inference.py
```

### Step 8B.4: Deploy Daily Scheduled Infrastructure (CDK)

**Architecture:** `EventBridge (daily cron) → Lambda → SageMaker Processing Job → S3 predictions`

1. Copy infrastructure: `cp -r <SKILL_DIR>/infrastructure ./infrastructure`
2. Configure `infrastructure/cdk.json` (input_bucket, ecr_image_uri, model_s3_uri, schedule)
3. Upload batch code to S3:
   ```bash
   aws s3 sync ./pdm/ s3://${OUTPUT_BUCKET}/batch-code/pdm/
   aws s3 cp batch_inference.py s3://${OUTPUT_BUCKET}/batch-code/
   aws s3 cp fault_prediction/baseline/runtime.py s3://${OUTPUT_BUCKET}/batch-code/fault_prediction/baseline/runtime.py
   ```
4. Deploy: `cd infrastructure/ && pip install -r requirements.txt && cdk deploy`
5. Test: `aws lambda invoke --function-name pdm-batch-trigger /dev/stdout`

### Batch Inference Key Details

- **Output format**: CSV with entity ID + `{label}_proba` columns
- **Local runtime**: ~2–3 min (1000 devices, 7-day window, 6 labels)
- **Processing Job runtime**: ~5–7 min (includes instance spin-up)
- **Instance type**: `ml.m5.2xlarge` (32GB) for multi-label ensembles
- **No persistent infra**: compute terminates after job completes

## Error Recovery

| Failure | Recovery |
|---------|----------|
| `ModuleNotFoundError` in container | Add missing dep to Dockerfile, rebuild and push image |
| `OSError: libgomp.so` | Add `apt-get install libgomp1` to Dockerfile |
| `AssertionError: Python version` | Match Dockerfile `FROM python:X.Y-slim` to training env |
| Worker OOM / dies (AutoGluon DLC) | Use custom container, not DLC |
| `pickle` error in `load_child` | Use full `ag_model/` dir (not `clone_for_deployment`) |
| Invocation timeout on first call | Normal cold start — retry after 30s |
| Batch transform `Failed` — timeout | Increase `InvocationsTimeoutInSeconds`, reduce `MaxPayloadInMB` |
| Incomplete output (missing files) | Add S3 lifecycle rule to abort incomplete multipart uploads |
| `NoSuchBucket` on predictions bucket | `ensure_output_bucket()` failed — check IAM `s3:CreateBucket` permission |
| Only 1 instance active despite `InstanceCount>1` | Split input into multiple files — 1 file = 1 instance max |
| CSV parsing error in container | Input has embedded newlines — clean data or use JSON Lines format |
| Processing Job OOM (`use an instance type with more memory`) | Model + telemetry exceeds 16GB — use `ml.m5.2xlarge` (32GB) |
| Processing Job `ModuleNotFoundError: pdm` | Code not uploaded to S3 — re-run `aws s3 sync ./pdm/ s3://<output-bucket>/batch-code/pdm/` |
