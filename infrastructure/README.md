# PdM Batch Inference Infrastructure (CDK)

Deploys a daily batch inference pipeline:

```
EventBridge (cron) → Lambda → SageMaker Processing Job → S3 predictions
```

## Prerequisites

- AWS CDK CLI installed (`npm install -g aws-cdk`)
- Docker image pushed to ECR (from Phase 8A/8B container build)
- Model artifacts uploaded to S3 (from Phase 7)
- Batch inference code uploaded to `s3://{input_bucket}/batch-code/`

## Configuration

Edit `cdk.json` context variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `project_name` | Prefix for all resource names | `pdm` |
| `input_bucket` | S3 bucket with telemetry data (read-only) | `my-telemetry-bucket` |
| `ecr_image_uri` | Full ECR image URI | `123456.dkr.ecr.eu-central-1.amazonaws.com/pdm-inference:latest` |
| `model_s3_uri` | S3 path to model artifacts (in output bucket) | `s3://my-output-bucket/batch-model/` |
| `instance_type` | SageMaker instance type | `ml.m5.2xlarge` |
| `schedule` | EventBridge cron expression | `cron(0 6 * * ? *)` (daily at 06:00 UTC) |
| `lookback_days` | Telemetry window size | `7` |

## Deploy

```bash
cd infrastructure/
pip install -r requirements.txt
cdk bootstrap  # first time only
cdk deploy -c account=$(aws sts get-caller-identity --query Account --output text) \
           -c region=${AWS_REGION:-eu-central-1}
```

## Upload Batch Code

Before the Processing Job can run, upload the batch inference script and pdm library to the **output bucket** (never write to the input bucket):

```bash
# From the project root:
aws s3 sync ./pdm/ s3://${OUTPUT_BUCKET}/batch-code/pdm/
aws s3 cp batch_inference.py s3://${OUTPUT_BUCKET}/batch-code/
aws s3 cp fault_prediction/baseline/runtime.py s3://${OUTPUT_BUCKET}/batch-code/fault_prediction/baseline/runtime.py
```

## Test Manually

Invoke the Lambda to trigger a Processing Job immediately:

```bash
aws lambda invoke --function-name pdm-batch-trigger /dev/stdout
```

## Architecture

- **EventBridge**: Fires daily at configured time
- **Lambda**: Lightweight trigger (~20 lines) that calls `create_processing_job`
- **SageMaker Processing Job**: Runs the full batch script inside the container
  - Reads telemetry from S3
  - Loads model from S3
  - Writes predictions to `s3://{input_bucket}-predictions/{YYYYMMDD}/`
- **Container**: Same Docker image used for the real-time endpoint
