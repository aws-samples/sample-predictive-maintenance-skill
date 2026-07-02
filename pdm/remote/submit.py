"""Submit a single training job to SageMaker.

Usage:
    uv run python pdm/remote/submit.py --train data/raw_train.csv --test data/raw_test.csv \
        --formulation rul --window-size 30 --optuna --n-trials 100
"""
import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import boto3


def _log(msg):
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_execution_role(region: str) -> str:
    """Find a SageMaker execution role in the account."""
    iam = boto3.client("iam", region_name=region)
    for page in iam.get_paginator("list_roles").paginate():
        for role in page["Roles"]:
            if "SageMaker" in role["RoleName"] and "Execution" in role["RoleName"]:
                return role["Arn"]
    account = boto3.client("sts", region_name=region).get_caller_identity()["Account"]
    return f"arn:aws:iam::{account}:role/service-role/AmazonSageMaker-ExecutionRole"


def upload_data(train_path: Path, test_path: Path, bucket: str, prefix: str, region: str) -> str:
    """Upload train/test CSVs to S3, return s3 URI."""
    s3 = boto3.client("s3", region_name=region)
    s3_prefix = f"{prefix}/data"
    for f in [train_path, test_path]:
        key = f"{s3_prefix}/{f.name}"
        s3.upload_file(str(f), bucket, key)
        _log(f"  ↑ {f.name} → s3://{bucket}/{key}")
    return f"s3://{bucket}/{s3_prefix}"


def package_source(skill_dir: Path) -> str:
    """Package pdm/ source + train_remote.py entry script into a temp dir."""
    tmp = tempfile.mkdtemp(prefix="pdm_sm_src_")
    dest = Path(tmp)
    shutil.copytree(skill_dir / "pdm", dest / "pdm",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "remote"))
    # Copy the remote entry script
    shutil.copy2(skill_dir / "pdm" / "remote" / "train_remote.py", dest / "train_remote.py")
    # Requirements for container
    (dest / "requirements.txt").write_text(
        "autogluon.tabular[lightgbm,catboost,xgboost]>=1.2\noptuna>=3.0\n"
        "lifelines>=0.29\nray>=2.10.0,<2.45.0\n"
    )
    total = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    _log(f"Source package: {total / 1024:.0f} KB")
    return str(dest)


def submit_training_job(
    train_path: Path, test_path: Path,
    hyperparameters: dict,
    instance_type: str = "ml.m5.4xlarge",
    region: str = None,
    spot: bool = True,
    job_name_prefix: str = "pdm",
    wait: bool = True,
) -> dict:
    """Submit a SageMaker training job.

    Returns dict with job_name, status, metrics, model_artifact_uri.
    """
    import os
    region = region or os.environ.get("AWS_REGION", "eu-west-1")
    session = boto3.session.Session(region_name=region)
    sm = session.client("sagemaker")
    s3 = session.client("s3")

    # Resolve bucket and role
    sts = session.client("sts")
    account = sts.get_caller_identity()["Account"]
    bucket = f"sagemaker-{region}-{account}"
    role = get_execution_role(region)

    # Ensure bucket exists
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        _log(f"Creating bucket: {bucket}")
        create_args = {"Bucket": bucket}
        if region != "us-east-1":
            create_args["CreateBucketConfiguration"] = {"LocationConstraint": region}
        s3.create_bucket(**create_args)

    timestamp = time.strftime("%Y%m%d%H%M%S")
    job_name = f"{job_name_prefix}-{timestamp}"
    prefix = f"pdm-training/{job_name}"

    _log(f"Job: {job_name}")
    _log(f"Instance: {instance_type} {'(Spot)' if spot else ''}")
    _log(f"Role: {role}")

    # Upload data
    s3_data = upload_data(train_path, test_path, bucket, prefix, region)

    # Package source
    skill_dir = Path(__file__).parent.parent.parent
    source_dir = package_source(skill_dir)

    # Upload source as tar.gz
    import tarfile
    fd, _tmp = tempfile.mkstemp(suffix=".tar.gz")
    os.close(fd)
    source_tar = Path(_tmp)
    with tarfile.open(source_tar, "w:gz") as tar:
        tar.add(source_dir, arcname=".")
    s3_source_key = f"{prefix}/source/source.tar.gz"
    s3.upload_file(str(source_tar), bucket, s3_source_key)
    source_tar.unlink()
    shutil.rmtree(source_dir)
    _log(f"Source uploaded: s3://{bucket}/{s3_source_key}")

    # Training image (sklearn container with pip install support)
    DLC_REGISTRIES = {
        "us-east-1": "683313688378", "us-east-2": "257758044811",
        "us-west-1": "746614075791", "us-west-2": "246618743249",
        "eu-west-1": "141502667606", "eu-west-2": "764974769150",
        "eu-central-1": "492215442770", "eu-north-1": "662702820516",
        "ap-northeast-1": "354813040037", "ap-southeast-1": "121021644041",
        "ap-southeast-2": "783357654285",
    }
    ecr_account = DLC_REGISTRIES.get(region, "683313688378")
    image_uri = f"{ecr_account}.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.4-2-py312-cpu-py3"

    # Build training job config
    hp_strings = {k: str(v) for k, v in hyperparameters.items()}

    training_params = {
        "TrainingJobName": job_name,
        "RoleArn": role,
        "AlgorithmSpecification": {
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
        },
        "HyperParameters": {
            **hp_strings,
            "sagemaker_program": "train_remote.py",
            "sagemaker_submit_directory": f"s3://{bucket}/{s3_source_key}",
        },
        "InputDataConfig": [{
            "ChannelName": "train",
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": s3_data}},
        }],
        "OutputDataConfig": {"S3OutputPath": f"s3://{bucket}/{prefix}/output"},
        "ResourceConfig": {
            "InstanceType": instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
    }

    if spot:
        training_params["EnableManagedSpotTraining"] = True
        training_params["StoppingCondition"]["MaxWaitTimeInSeconds"] = 7200

    sm.create_training_job(**training_params)
    _log("Job submitted")

    if not wait:
        return {"job_name": job_name, "status": "Submitted"}

    # Poll until complete
    return poll_job(sm, job_name)


def poll_job(sm_client, job_name: str) -> dict:
    """Poll a training job until completion, return results."""
    last_status = None
    while True:
        desc = sm_client.describe_training_job(TrainingJobName=job_name)
        status = desc["TrainingJobStatus"]
        secondary = desc.get("SecondaryStatus", "")
        msg = f"{status}/{secondary}"
        if msg != last_status:
            _log(msg)
            last_status = msg
        if status == "Completed":
            model_uri = desc["ModelArtifacts"]["S3ModelArtifacts"]
            metrics = {}
            for m in desc.get("FinalMetricDataList", []):
                metrics[m["MetricName"]] = m["Value"]
            _log(f"✅ Complete — model: {model_uri}")
            return {"job_name": job_name, "status": "Completed",
                    "model_uri": model_uri, "metrics": metrics}
        if status in ("Failed", "Stopped"):
            reason = desc.get("FailureReason", "unknown")
            _log(f"❌ {status}: {reason}")
            return {"job_name": job_name, "status": status, "reason": reason}
        time.sleep(30)


def fetch_model(model_uri: str, output_dir: Path, region: str = None) -> Path:
    """Download and extract model artifacts from S3 to a local directory.

    Args:
        model_uri: S3 URI to model.tar.gz (returned by submit_training_job)
        output_dir: Local directory to extract into
        region: AWS region

    Returns:
        Path to the extracted model directory containing metrics.json + model files.
    """
    import os
    import tarfile
    region = region or os.environ.get("AWS_REGION", "eu-west-1")
    s3 = boto3.client("s3", region_name=region)

    # Parse s3://bucket/key
    parts = model_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tar_path = output_dir / "model.tar.gz"

    _log(f"Downloading {model_uri}...")
    s3.download_file(bucket, key, str(tar_path))
    _log(f"Extracting to {output_dir}/")
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.startswith("/") or ".." in member.name.split("/"):
                raise ValueError(f"Unsafe tar member: {member.name}")
            target = output_dir / member.name
            if not target.resolve().is_relative_to(output_dir.resolve()):
                raise ValueError(f"Unsafe tar member: {member.name}")
        for member in tar.getmembers():
            tar.extract(member, output_dir, filter="data")  # nosec
    tar_path.unlink()

    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        _log(f"✅ Metrics: {json.dumps(metrics)}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="SageMaker training job management")
    sub = parser.add_subparsers(dest="command")

    # Submit command
    submit_p = sub.add_parser("submit", help="Submit a training job")
    submit_p.add_argument("--train", type=Path, required=True)
    submit_p.add_argument("--test", type=Path, required=True)
    submit_p.add_argument("--formulation", required=True, choices=["rul", "classification", "survival"])
    submit_p.add_argument("--window-size", type=int, default=30)
    submit_p.add_argument("--rul-cap", type=int, default=125)
    submit_p.add_argument("--optuna", action="store_true")
    submit_p.add_argument("--n-trials", type=int, default=50)
    submit_p.add_argument("--time-limit", type=int, default=300)
    submit_p.add_argument("--instance-type", default="ml.m5.4xlarge")
    submit_p.add_argument("--no-spot", action="store_true")
    submit_p.add_argument("--no-wait", action="store_true")
    submit_p.add_argument("--region", default=None)

    # Fetch command
    fetch_p = sub.add_parser("fetch", help="Download model artifacts from a completed job")
    fetch_p.add_argument("--job-name", required=True, help="SageMaker training job name")
    fetch_p.add_argument("--output", type=Path, default=Path("./model"), help="Local output directory")
    fetch_p.add_argument("--region", default=None)

    args = parser.parse_args()

    if args.command == "fetch":
        import os
        region = args.region or os.environ.get("AWS_REGION", "eu-west-1")
        sm = boto3.client("sagemaker", region_name=region)
        desc = sm.describe_training_job(TrainingJobName=args.job_name)
        if desc["TrainingJobStatus"] != "Completed":
            print(f"Job {args.job_name} is {desc['TrainingJobStatus']}, cannot fetch.")
            return
        model_uri = desc["ModelArtifacts"]["S3ModelArtifacts"]
        fetch_model(model_uri, args.output, region)

    elif args.command == "submit":
        hp = {
            "formulation": args.formulation,
            "window-size": args.window_size,
            "rul-cap": args.rul_cap,
            "time-limit": args.time_limit,
            "backend": "optuna" if args.optuna else "autogluon",
            "n-trials": args.n_trials,
        }
        result = submit_training_job(
            train_path=args.train, test_path=args.test,
            hyperparameters=hp,
            instance_type=args.instance_type,
            region=args.region,
            spot=not args.no_spot,
            wait=not args.no_wait,
        )
        print(json.dumps(result, indent=2))
        # Auto-fetch if job completed
        if result.get("status") == "Completed" and result.get("model_uri"):
            fetch_model(result["model_uri"], Path("./model"), args.region)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
