"""Lambda trigger for PdM batch inference SageMaker Processing Job.

Invoked daily by EventBridge. Creates a SageMaker Processing Job that:
1. Pulls the custom container (same image as endpoint)
2. Runs batch_inference.py inside the container
3. Reads telemetry from S3 input bucket, writes predictions to output bucket

Environment variables (set by CDK stack):
- PROJECT_NAME: Project identifier for naming
- INPUT_BUCKET: S3 bucket with telemetry data
- OUTPUT_BUCKET: S3 bucket for predictions output
- ECR_IMAGE_URI: Docker image URI
- MODEL_S3_URI: s3:// path to model.tar.gz (or model directory)
- SAGEMAKER_ROLE_ARN: IAM role for the Processing Job
- INSTANCE_TYPE: ML instance type (default: ml.m5.xlarge)
- LOOKBACK_DAYS: Telemetry window size (default: 7)
"""
import os
import time
import boto3


def handler(event, context):
    sm = boto3.client("sagemaker")

    project_name = os.environ["PROJECT_NAME"]
    input_bucket = os.environ["INPUT_BUCKET"]
    output_bucket = os.environ["OUTPUT_BUCKET"]
    ecr_image_uri = os.environ["ECR_IMAGE_URI"]
    model_s3_uri = os.environ["MODEL_S3_URI"]
    role_arn = os.environ["SAGEMAKER_ROLE_ARN"]
    instance_type = os.environ.get("INSTANCE_TYPE", "ml.m5.xlarge")
    lookback_days = os.environ.get("LOOKBACK_DAYS", "7")

    job_name = f"{project_name}-batch-{int(time.time())}"

    sm.create_processing_job(
        ProcessingJobName=job_name,
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": instance_type,
                "VolumeSizeInGB": 30,
            }
        },
        AppSpecification={
            "ImageUri": ecr_image_uri,
            "ContainerEntrypoint": ["python", "-u", "/opt/ml/processing/code/batch_inference.py"],
        },
        Environment={
            "INPUT_BUCKET": input_bucket,
            "OUTPUT_BUCKET": output_bucket,
            "MODEL_S3_URI": model_s3_uri,
            "LOOKBACK_DAYS": lookback_days,
        },
        ProcessingInputs=[
            {
                "InputName": "code",
                "S3Input": {
                    "S3Uri": f"s3://{output_bucket}/batch-code/",
                    "LocalPath": "/opt/ml/processing/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "model",
                "S3Input": {
                    "S3Uri": model_s3_uri,
                    "LocalPath": "/opt/ml/processing/model",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
        ],
        ProcessingOutputConfig={
            "Outputs": [
                {
                    "OutputName": "predictions",
                    "S3Output": {
                        "S3Uri": f"s3://{output_bucket}/",
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ]
        },
        RoleArn=role_arn,
        StoppingCondition={"MaxRuntimeInSeconds": 900},
    )

    print(f"Started Processing Job: {job_name}")
    return {"statusCode": 200, "jobName": job_name}
