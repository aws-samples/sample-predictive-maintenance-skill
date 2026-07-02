#!/usr/bin/env python3
"""Dispatch & monitor PdM experiments in parallel on SageMaker.

Supports all formulations: RUL, classification, multi-label, survival.
Provides real-time progress reporting with spot interruption tracking,
per-label/metric progress via CloudWatch logs, and ETA calculation.

Usage:
    # Submit multi-label experiments (pre-processed CSVs per experiment):
    uv run python pdm/remote/parallel.py \
        --experiments experiments.json --formulation multilabel \
        --instance-type ml.m5.2xlarge

    # Submit RUL/single-model experiments (shared raw data, different HPs):
    uv run python pdm/remote/parallel.py \
        --train data/raw_train.csv --test data/raw_test.csv \
        --experiments experiments.json --formulation rul \
        --instance-type ml.m5.4xlarge

    # Monitor already-running jobs:
    uv run python pdm/remote/parallel.py --monitor

    # Live watch with log streaming (refreshes every 30s):
    uv run python pdm/remote/parallel.py --monitor --logs --watch

experiments.json format (multi-label — each experiment has its own data):
[
    {"name": "01_interactions", "train": "experiments/01/data/train.csv", "test": "experiments/01/data/test.csv", "time-limit": 120},
    {"name": "02_cross_signals", "train": "experiments/02/data/train.csv", "test": "experiments/02/data/test.csv", "time-limit": 120}
]

experiments.json format (RUL/classification — shared data, different hyperparameters):
[
    {"name": "baseline", "window-size": 30, "backend": "optuna", "n-trials": 100},
    {"name": "window20", "window-size": 20, "backend": "optuna", "n-trials": 100}
]
"""
import argparse
import json
import os
import re
import shutil
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import boto3

# ─── Constants ────────────────────────────────────────────────────────────────

LOG_GROUP = "/aws/sagemaker/TrainingJobs"
LABEL_PATTERN = re.compile(r"\[(\d+)/(\d+)\] Training (label_\S+)")
RESULT_PATTERN = re.compile(r"(label_\S+): F1=([\d.]+) P=([\d.]+) R=([\d.]+)")
METRIC_PATTERN = re.compile(r"✅ Metrics: (.+)")
JOB_PREFIX = "pdm-"


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class LabelResult:
    name: str
    f1: float
    precision: float
    recall: float


@dataclass
class JobProgress:
    name: str
    job_name: str
    status: str = "Pending"
    secondary: str = ""
    interruptions: int = 0
    current_label_idx: int = 0
    total_labels: int = 0
    labels_completed: list = field(default_factory=list)
    training_start: datetime | None = None
    avg_label_time_s: float | None = None
    model_uri: str | None = None
    failure_reason: str | None = None
    log_tail: list = field(default_factory=list)
    final_metrics: dict = field(default_factory=dict)

    @property
    def eta_s(self) -> float | None:
        if not self.avg_label_time_s or not self.labels_completed:
            return None
        remaining = self.total_labels - len(self.labels_completed)
        return remaining * self.avg_label_time_s

    @property
    def progress_pct(self) -> float:
        if not self.total_labels:
            return 0
        return len(self.labels_completed) / self.total_labels * 100


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _ts():
    return time.strftime("%H:%M:%S")


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def _status_icon(status: str, secondary: str) -> str:
    if status == "Completed":
        return "✅"
    if status in ("Failed", "Stopped"):
        return "❌"
    if secondary == "Interrupted":
        return "⚡"
    if secondary == "Training":
        return "🏋️"
    if secondary in ("Starting", "Downloading"):
        return "⏳"
    return "⏸️"


def _progress_bar(jp: JobProgress, width: int = 20) -> str:
    filled = int(jp.progress_pct / 100 * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {jp.progress_pct:.0f}%"


# ─── Log Parsing ──────────────────────────────────────────────────────────────

def _parse_training_logs(logs_client, job_name: str, progress: JobProgress, stream_logs: bool):
    """Parse CloudWatch logs for per-label progress and metrics."""
    try:
        resp = logs_client.describe_log_streams(
            logGroupName=LOG_GROUP,
            logStreamNamePrefix=f"{job_name}/",
            orderBy="LogStreamName", descending=True, limit=5,
        )
    except Exception:
        return

    streams = resp.get("logStreams", [])
    if not streams:
        return

    # Use stream with most recent activity
    streams.sort(key=lambda s: s.get("lastEventTimestamp", 0), reverse=True)
    stream_name = streams[0]["logStreamName"]

    try:
        resp = logs_client.get_log_events(
            logGroupName=LOG_GROUP, logStreamName=stream_name,
            startFromHead=True, limit=500,
        )
    except Exception:
        return

    all_events = resp.get("events", [])
    for event in all_events:
        msg = event["message"]
        m = LABEL_PATTERN.search(msg)
        if m:
            progress.current_label_idx = int(m.group(1))
            progress.total_labels = int(m.group(2))

        m = RESULT_PATTERN.search(msg)
        if m:
            result = LabelResult(m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4)))
            if not any(lr.name == result.name for lr in progress.labels_completed):
                progress.labels_completed.append(result)

        m = METRIC_PATTERN.search(msg)
        if m:
            try:
                # Parse "mean_f1=0.8187, median_f1=0.8984" format
                for part in m.group(1).split(","):
                    k, v = part.strip().split("=")
                    progress.final_metrics[k] = float(v)
            except Exception:
                pass

    # Capture filtered log tail
    if stream_logs:
        skip = {"Failed to import torch", "sagemaker-training-toolkit INFO",
                "Returning the value itself", "Failed to parse hyperparameter",
                "UserWarning", "stacked_overfitting"}
        for event in all_events[-10:]:
            msg = event["message"].strip().replace("#011", "  ")
            if msg and not any(s in msg for s in skip):
                progress.log_tail.append(msg)


# ─── Job Progress ─────────────────────────────────────────────────────────────

def get_job_progress(sm, logs_client, job_name: str, experiment_name: str, stream_logs: bool = False) -> JobProgress:
    """Build progress state from job description + CloudWatch logs."""
    desc = sm.describe_training_job(TrainingJobName=job_name)
    progress = JobProgress(name=experiment_name, job_name=job_name)
    progress.status = desc["TrainingJobStatus"]
    progress.secondary = desc.get("SecondaryStatus", "")
    progress.model_uri = desc.get("ModelArtifacts", {}).get("S3ModelArtifacts")
    progress.failure_reason = desc.get("FailureReason")

    transitions = desc.get("SecondaryStatusTransitions", [])
    progress.interruptions = sum(1 for t in transitions if t["Status"] == "Interrupted")

    for t in reversed(transitions):
        if t["Status"] == "Training":
            ts = t["StartTime"]
            progress.training_start = ts.replace(tzinfo=timezone.utc) if isinstance(ts, datetime) and ts.tzinfo is None else (ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts)))
            break

    _parse_training_logs(logs_client, job_name, progress, stream_logs)

    if progress.labels_completed and progress.training_start:
        elapsed = (datetime.now(timezone.utc) - progress.training_start).total_seconds()
        progress.avg_label_time_s = elapsed / len(progress.labels_completed)

    return progress


# ─── Display ──────────────────────────────────────────────────────────────────

def print_dashboard(jobs_progress: list[JobProgress]):
    """Print a rich progress dashboard."""
    print(f"\n{'━'*74}")
    print(f"  📊 EXPERIMENT DASHBOARD  ({_ts()})")
    print(f"{'━'*74}")

    for jp in jobs_progress:
        icon = _status_icon(jp.status, jp.secondary)
        eta = _format_duration(jp.eta_s)

        print(f"\n  {icon} {jp.name}")
        print(f"    Status: {jp.status}/{jp.secondary}", end="")
        if jp.interruptions:
            print(f"  ⚡ {jp.interruptions} spot interruption{'s' if jp.interruptions > 1 else ''}", end="")
        print()

        if jp.total_labels:
            bar = _progress_bar(jp)
            print(f"    Progress: {bar}  {len(jp.labels_completed)}/{jp.total_labels} labels")
            if jp.labels_completed:
                print(f"    ETA: {eta}  |  Avg: {_format_duration(jp.avg_label_time_s)}/label")
                print(f"    ┌{'─'*58}┐")
                print(f"    │ {'Label':<35} {'F1':>6} {'Prec':>6} {'Rec':>6} │")
                print(f"    ├{'─'*58}┤")
                for lr in jp.labels_completed:
                    print(f"    │ {lr.name:<35} {lr.f1:>6.4f} {lr.precision:>6.4f} {lr.recall:>6.4f} │")
                if jp.current_label_idx > len(jp.labels_completed):
                    print(f"    │ {'(training...)':<35} {'—':>6} {'—':>6} {'—':>6} │")
                print(f"    └{'─'*58}┘")
        elif jp.final_metrics:
            print(f"    Metrics: {jp.final_metrics}")

        if jp.failure_reason:
            print(f"    ❌ {jp.failure_reason[:80]}")
        if jp.log_tail:
            print(f"    Log:")
            for line in jp.log_tail[-5:]:
                print(f"    │ {line[:72]}")

    print(f"\n{'━'*74}")
    completed = [jp for jp in jobs_progress if jp.status == "Completed"]
    in_progress = [jp for jp in jobs_progress if jp.status == "InProgress"]
    total_interruptions = sum(jp.interruptions for jp in jobs_progress)
    parts = []
    if completed:
        parts.append(f"✅ {len(completed)} done")
    if in_progress:
        parts.append(f"🏋️ {len(in_progress)} running")
    if total_interruptions:
        parts.append(f"⚡ {total_interruptions} total interruptions")
    print(f"  {' | '.join(parts)}")
    print(f"{'━'*74}\n")


def _print_final_comparison(jobs_progress: list[JobProgress]):
    """Print final comparison table."""
    completed = [jp for jp in jobs_progress if jp.status == "Completed" and jp.labels_completed]
    if not completed:
        return

    print(f"\n{'='*74}")
    print("  🏆 FINAL COMPARISON")
    print(f"{'='*74}")

    results = []
    for jp in completed:
        f1s = [lr.f1 for lr in jp.labels_completed]
        results.append((jp.name, sum(f1s) / len(f1s), sorted(f1s)[len(f1s) // 2], jp.labels_completed))
    results.sort(key=lambda r: -r[1])

    for i, (name, mean_f1, median_f1, labels) in enumerate(results):
        marker = "👑" if i == 0 else "  "
        print(f"\n  {marker} {name}: mean_f1={mean_f1:.4f}  median_f1={median_f1:.4f}")
        for lr in sorted(labels, key=lambda x: -x.f1):
            print(f"       {lr.name:<35} F1={lr.f1:.4f}")

    print(f"\n{'='*74}")
    out = Path("parallel_results.json")
    data = [{"name": n, "mean_f1": m, "median_f1": md,
             "labels": {lr.name: {"f1": lr.f1, "precision": lr.precision, "recall": lr.recall} for lr in ls}}
            for n, m, md, ls in results]
    out.write_text(json.dumps(data, indent=2))
    print(f"  Results saved to {out}")


# ─── Commands ─────────────────────────────────────────────────────────────────

def discover_jobs(sm, prefix: str = "pdm-") -> list[dict]:
    """Find running or recent pdm training jobs."""
    jobs = []
    for status in ["InProgress", "Completed", "Failed", "Stopped"]:
        resp = sm.list_training_jobs(
            NameContains=prefix, StatusEquals=status,
            SortBy="CreationTime", SortOrder="Descending", MaxResults=10,
        )
        jobs.extend(resp.get("TrainingJobSummaries", []))
    seen = set()
    return [j for j in jobs if not (j["TrainingJobName"] in seen or seen.add(j["TrainingJobName"]))]


def group_by_batch(jobs: list[dict]) -> dict[str, list[dict]]:
    """Group jobs by dispatch timestamp."""
    batches = {}
    for j in jobs:
        parts = j["TrainingJobName"].rsplit("-", 1)
        ts = parts[-1] if len(parts) == 2 and parts[-1].isdigit() else "unknown"
        batches.setdefault(ts, []).append(j)
    return batches


def cmd_monitor(region: str, stream_logs: bool, watch: bool, prefix: str = "pdm-"):
    """Monitor existing training jobs."""
    session = boto3.session.Session(region_name=region)
    sm = session.client("sagemaker")
    logs_client = session.client("logs")

    while True:
        all_jobs = discover_jobs(sm, prefix)
        if not all_jobs:
            print("No training jobs found.")
            return

        batches = group_by_batch(all_jobs)
        latest_ts = sorted(batches.keys(), reverse=True)[0]
        batch_jobs = batches[latest_ts]

        progress_list = []
        for j in batch_jobs:
            name = j["TrainingJobName"].replace(f"-{latest_ts}", "").replace("pdm-fp-", "").replace("pdm-", "").replace("-", "_")
            jp = get_job_progress(sm, logs_client, j["TrainingJobName"], name, stream_logs)
            progress_list.append(jp)

        progress_list.sort(key=lambda p: (p.status != "InProgress", p.name))
        print_dashboard(progress_list)

        all_done = all(jp.status in ("Completed", "Failed", "Stopped") for jp in progress_list)
        if all_done:
            _print_final_comparison(progress_list)
            return
        if not watch:
            return
        time.sleep(30)
        print("\033[2J\033[H", end="")


# ─── Submit ───────────────────────────────────────────────────────────────────

def get_execution_role(region: str) -> str:
    iam = boto3.client("iam", region_name=region)
    for page in iam.get_paginator("list_roles").paginate():
        for role in page["Roles"]:
            if "SageMaker" in role["RoleName"] and "Execution" in role["RoleName"]:
                return role["Arn"]
    account = boto3.client("sts", region_name=region).get_caller_identity()["Account"]
    return f"arn:aws:iam::{account}:role/service-role/AmazonSageMaker-ExecutionRole"


def package_source(project_dir: Path, formulation: str) -> Path:
    """Package pdm/ + entry script into a source tarball."""
    tmp = Path(tempfile.mkdtemp(prefix="pdm_src_"))
    shutil.copytree(project_dir / "pdm", tmp / "pdm",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "remote"))
    # Copy the appropriate entry script
    entry = "train_multilabel.py" if formulation == "multilabel" else "train_remote.py"
    shutil.copy2(project_dir / "pdm" / "remote" / entry, tmp / entry)
    (tmp / "requirements.txt").write_text(
        "autogluon.tabular[lightgbm,catboost,xgboost]>=1.2\n"
        "lifelines>=0.29\noptuna>=3.0\nray>=2.10.0,<2.45.0\n"
    )
    fd, _tmp = tempfile.mkstemp(suffix=".tar.gz")
    os.close(fd)
    tar_path = Path(_tmp)
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(str(tmp), arcname=".")
    shutil.rmtree(tmp)
    return tar_path


def cmd_submit(region: str, formulation: str, experiments_path: Path,
               train_path: Path | None, test_path: Path | None,
               instance_type: str, spot: bool):
    """Submit experiments and start monitoring."""
    import os
    session = boto3.session.Session(region_name=region)
    sm = session.client("sagemaker")
    s3 = session.client("s3")
    account = session.client("sts").get_caller_identity()["Account"]
    bucket = f"sagemaker-{region}-{account}"
    role = get_execution_role(region)

    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        create_args = {"Bucket": bucket}
        if region != "us-east-1":
            create_args["CreateBucketConfiguration"] = {"LocationConstraint": region}
        s3.create_bucket(**create_args)

    timestamp = time.strftime("%Y%m%d%H%M%S")
    prefix = f"pdm-experiments/{timestamp}"
    experiments = json.loads(experiments_path.read_text())
    project_dir = Path.cwd()

    print(f"[{_ts()}] 📦 Packaging source...")
    source_tar = package_source(project_dir, formulation)
    s3_source_key = f"{prefix}/source/source.tar.gz"
    s3.upload_file(str(source_tar), bucket, s3_source_key)
    source_tar.unlink()

    DLC_REGISTRIES = {
        "us-east-1": "683313688378", "us-east-2": "257758044811",
        "us-west-2": "246618743249", "eu-west-1": "141502667606",
        "eu-west-2": "764974769150", "eu-central-1": "492215442770",
    }
    ecr_account = DLC_REGISTRIES.get(region, "683313688378")
    image_uri = f"{ecr_account}.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.4-2-py312-cpu-py3"

    # Upload shared data (for non-multilabel)
    shared_data_uri = None
    if train_path and test_path:
        s3_data_prefix = f"{prefix}/data/shared"
        s3.upload_file(str(train_path), bucket, f"{s3_data_prefix}/raw_train.csv")
        s3.upload_file(str(test_path), bucket, f"{s3_data_prefix}/raw_test.csv")
        shared_data_uri = f"s3://{bucket}/{s3_data_prefix}"

    spot_msg = " (spot)" if spot else " (on-demand)"
    print(f"[{_ts()}] 🚀 Submitting {len(experiments)} experiments on {instance_type}{spot_msg}")
    if spot:
        print(f"[{_ts()}] ⚠️  Spot saves ~70% but risks interruption + restart. Use --no-spot for guaranteed capacity.")

    entry_script = "train_multilabel.py" if formulation == "multilabel" else "train_remote.py"
    job_prefix = "pdm-fp" if formulation == "multilabel" else "pdm"

    for exp in experiments:
        name = exp.get("name", f"exp_{len(experiments)}")

        # Determine data URI
        if formulation == "multilabel":
            # Each experiment has its own pre-processed data
            exp_train, exp_test = Path(exp["train"]), Path(exp["test"])
            s3_data_prefix = f"{prefix}/data/{name}"
            s3.upload_file(str(exp_train), bucket, f"{s3_data_prefix}/train.csv")
            s3.upload_file(str(exp_test), bucket, f"{s3_data_prefix}/test.csv")
            data_uri = f"s3://{bucket}/{s3_data_prefix}"
        else:
            data_uri = shared_data_uri

        job_name = f"{job_prefix}-{name.replace('_', '-')}-{timestamp}"

        # Build hyperparameters
        hp = {"sagemaker_program": entry_script,
              "sagemaker_submit_directory": f"s3://{bucket}/{s3_source_key}"}
        if formulation == "multilabel":
            hp.update({"experiment-name": name, "time-limit": str(exp.get("time-limit", 120)),
                       "presets": exp.get("presets", "best")})
        else:
            hp.update({"formulation": formulation,
                       "window-size": str(exp.get("window-size", 30)),
                       "rul-cap": str(exp.get("rul-cap", 125)),
                       "time-limit": str(exp.get("time-limit", 300)),
                       "backend": exp.get("backend", "optuna"),
                       "n-trials": str(exp.get("n-trials", 50))})

        training_params = {
            "TrainingJobName": job_name, "RoleArn": role,
            "AlgorithmSpecification": {"TrainingImage": image_uri, "TrainingInputMode": "File"},
            "HyperParameters": hp,
            "InputDataConfig": [{"ChannelName": "train",
                                 "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": data_uri}}}],
            "OutputDataConfig": {"S3OutputPath": f"s3://{bucket}/{prefix}/output/{name}"},
            "ResourceConfig": {"InstanceType": instance_type, "InstanceCount": 1, "VolumeSizeInGB": 30},
            "StoppingCondition": {"MaxRuntimeInSeconds": 7200},
        }
        if spot:
            training_params["EnableManagedSpotTraining"] = True
            training_params["StoppingCondition"]["MaxWaitTimeInSeconds"] = 10800

        sm.create_training_job(**training_params)
        print(f"[{_ts()}]   ✓ {name} → {job_name}")

    print(f"\n[{_ts()}] 📊 All submitted. Starting live monitoring...\n")
    time.sleep(5)
    cmd_monitor(region, stream_logs=False, watch=True, prefix=job_prefix)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dispatch & monitor SageMaker PdM experiments")
    parser.add_argument("--experiments", type=Path, help="experiments.json to submit")
    parser.add_argument("--train", type=Path, help="Shared training data (RUL/classification)")
    parser.add_argument("--test", type=Path, help="Shared test data (RUL/classification)")
    parser.add_argument("--formulation", choices=["rul", "classification", "survival", "multilabel"],
                        default="multilabel")
    parser.add_argument("--monitor", action="store_true", help="Monitor existing jobs")
    parser.add_argument("--logs", action="store_true", help="Stream training log lines")
    parser.add_argument("--watch", action="store_true", help="Refresh until all done")
    parser.add_argument("--instance-type", default="ml.m5.2xlarge")
    parser.add_argument("--no-spot", action="store_true")
    parser.add_argument("--region", default=None)
    args = parser.parse_args()

    import os
    region = args.region or os.environ.get("AWS_REGION", "eu-central-1")

    if args.monitor or not args.experiments:
        cmd_monitor(region, stream_logs=args.logs, watch=args.watch)
    else:
        cmd_submit(region, args.formulation, args.experiments,
                   args.train, args.test, args.instance_type, spot=not args.no_spot)


if __name__ == "__main__":
    main()
