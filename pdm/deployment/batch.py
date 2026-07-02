"""Reusable batch inference utilities for PdM models.

Provides:
- load_telemetry_window: Load telemetry partitions for a date range
- aggregate_telemetry: Per-device aggregation (mean/std/max/min)
- predict_proba: Multi-label probability-only prediction (no feature importance)
- score_anomalies: Isolation Forest scoring + threshold flagging
- explain_anomalies: Z-score feature deviation explanations

Performance notes (observed on ~1000 devices, 7-day window, 6 labels):
- Telemetry loading: ~3-5s per partition (S3 I/O bound), ~17s in Processing Job (network from VPC)
- Aggregation: <1s
- predict_proba per label: ~3-8s (AutoGluon ensemble inference)
- Total local: ~2-3 minutes | Processing Job: ~5-7 min (includes instance spin-up)
- Instance sizing: ml.m5.xlarge (16GB) may OOM with large models (>500MB) + telemetry in memory → use ml.m5.2xlarge (32GB)

Run with `python -u` to see tqdm progress bars with ETA in real time.
"""
import json
import os

import boto3
import pandas as pd
from autogluon.tabular import TabularPredictor


def load_telemetry_window(bucket: str, prefix: str, target_date: str, lookback_days: int, region: str) -> pd.DataFrame:
    """Load telemetry for a lookback window ending on target_date.

    Uses substring-based partition_filter (one call per day) since
    load_partitioned_parquet only accepts string filters, not callables.

    Args:
        bucket: S3 bucket name.
        prefix: Telemetry prefix (e.g. 'telemetry/').
        target_date: End date as 'YYYY-MM-DD'.
        lookback_days: Number of days in the window (inclusive of target_date).
        region: AWS region.
    """
    from pdm.data.data_exploration import load_partitioned_parquet
    from tqdm import tqdm

    end = pd.Timestamp(target_date)
    start = end - pd.Timedelta(days=lookback_days - 1)

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += pd.Timedelta(days=1)

    dfs = []
    for date_str in tqdm(dates, desc="Loading telemetry partitions"):
        df = load_partitioned_parquet(bucket, prefix, partition_filter=date_str)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        raise ValueError(f"No telemetry data for window {start.date()} to {end.date()}")
    return pd.concat(dfs, ignore_index=True)


def aggregate_telemetry(telemetry_df: pd.DataFrame, entity_col: str = "device_id") -> pd.DataFrame:
    """Aggregate telemetry per device: mean, std, max, min per sensor.

    Returns wide-format DataFrame with columns like 'sensor_name_stat'.
    """
    telemetry_df[entity_col] = telemetry_df[entity_col].astype(str)
    agg = telemetry_df.groupby([entity_col, "sensor_name"])["sample_value"].agg(
        ["mean", "std", "max", "min"]
    ).reset_index()

    pivoted = agg.pivot_table(index=entity_col, columns="sensor_name", values=["max", "mean", "min", "std"])
    pivoted.columns = [f"{sensor}_{stat}" for stat, sensor in pivoted.columns]
    pivoted = pivoted.reset_index()
    return pivoted


def predict_proba(model_dir: str, features_df: pd.DataFrame) -> pd.DataFrame:
    """Run multi-label prediction returning ONLY probabilities (no feature importance).

    This is optimized for speed: uses predict_proba directly without
    predict() or feature_importance() calls.

    Args:
        model_dir: Path to model directory containing metadata.json and ag_model/.
        features_df: DataFrame with engineered features.

    Returns:
        DataFrame with one column per label: '{label}_proba'.
    """
    from tqdm import tqdm

    with open(os.path.join(model_dir, "metadata.json")) as f:
        metadata = json.load(f)

    feature_names = metadata["feature_names"]
    X = features_df.reindex(columns=feature_names, fill_value=0)

    ag_path = os.path.join(model_dir, "ag_model")
    results = pd.DataFrame(index=features_df.index)

    for label in tqdm(metadata["label_names"], desc="Predicting labels"):
        label_dir = os.path.join(ag_path, label.replace(" ", "_"))
        if os.path.isdir(label_dir):
            predictor = TabularPredictor.load(label_dir, verbosity=0)
            proba = predictor.predict_proba(X)
            results[f"{label}_proba"] = proba.iloc[:, 1].values if proba.shape[1] == 2 else proba.iloc[:, 0].values

    return results


def score_anomalies(model_dir: str, features_df: pd.DataFrame) -> pd.DataFrame:
    """Score samples with Isolation Forest and flag anomalies.

    Args:
        model_dir: Path to AD model directory (must contain isolation_forest.joblib,
                   scaler.joblib, metadata.json, threshold.json).
        features_df: DataFrame with engineered features.

    Returns:
        DataFrame with columns: anomaly_score, is_anomaly.
    """
    import joblib
    import numpy as np

    metadata = json.loads(open(os.path.join(model_dir, "metadata.json")).read())
    threshold_data = json.loads(open(os.path.join(model_dir, "threshold.json")).read())
    model = joblib.load(os.path.join(model_dir, "isolation_forest.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))

    feature_names = metadata["feature_names"]
    threshold = threshold_data["threshold"]

    X = features_df.reindex(columns=feature_names, fill_value=0).fillna(0).values
    X_scaled = scaler.transform(X)
    scores = -model.score_samples(X_scaled)

    return pd.DataFrame({
        "anomaly_score": np.round(scores, 6),
        "is_anomaly": (scores > threshold).astype(int),
    }, index=features_df.index)


def explain_anomalies(model_dir: str, features_df: pd.DataFrame, top_k: int = 5) -> list[str]:
    """Explain anomalies via z-score deviation from training baseline.

    For each sample, computes how many standard deviations each feature is from
    the training mean, and returns the top-k most deviant features.

    Args:
        model_dir: Path to AD model directory (must contain metadata.json, baseline_stats.json).
        features_df: DataFrame with engineered features.
        top_k: Number of top deviating features to report per sample.

    Returns:
        List of explanation strings, one per sample, formatted as:
        "feature1(z=3.2); feature2(z=2.8); ..."
    """
    import numpy as np

    metadata = json.loads(open(os.path.join(model_dir, "metadata.json")).read())
    baseline_stats = json.loads(open(os.path.join(model_dir, "baseline_stats.json")).read())
    feature_names = metadata["feature_names"]

    X = features_df.reindex(columns=feature_names, fill_value=0).fillna(0).values

    explanations = []
    for i in range(len(X)):
        deviations = {}
        for j, fname in enumerate(feature_names):
            if fname in baseline_stats:
                stats = baseline_stats[fname]
                std = stats["std"] if stats["std"] > 0 else 1.0
                z = abs((X[i, j] - stats["mean"]) / std)
                deviations[fname] = round(z, 2)
        top = sorted(deviations.items(), key=lambda x: x[1], reverse=True)[:top_k]
        explanations.append("; ".join(f"{f}(z={z})" for f, z in top))
    return explanations


class BatchInferencePipeline:
    """Reusable batch inference pipeline template.

    Subclass and override `load_metadata()` and `engineer_features()` to customize
    for your project. The run() method handles the full pipeline:
    load telemetry → aggregate → join metadata → engineer → predict → upload.

    Usage:
        class MyPipeline(BatchInferencePipeline):
            def load_metadata(self):
                from pdm.data.utils import load_and_prepare_metadata
                from pdm.data.data_exploration import load_all_flat_parquet
                return load_and_prepare_metadata(
                    self.input_bucket, "device_master/", "device_id",
                    load_fn=lambda: load_all_flat_parquet(self.input_bucket, "device_master/"),
                    keep_cols=[...], date_cols_for_age=["manufactured_at_date"],
                )

            def engineer_features(self, df):
                from pdm.fault_prediction import baseline_engineer_features
                return baseline_engineer_features(df, drop_cols=["device_id"])

        pipeline = MyPipeline(input_bucket="my-bucket", output_bucket="my-predictions")
        pipeline.run("2026-04-26")
    """

    def __init__(
        self,
        input_bucket: str,
        output_bucket: str,
        telemetry_prefix: str = "telemetry/",
        entity_col: str = "device_id",
        model_dir: str | None = None,
        lookback_days: int = 7,
        region: str | None = None,
    ):
        self.input_bucket = input_bucket
        self.output_bucket = output_bucket
        self.telemetry_prefix = telemetry_prefix
        self.entity_col = entity_col
        self.lookback_days = lookback_days
        self.region = region or os.environ.get("AWS_REGION", "eu-central-1")

        # Auto-detect Processing Job environment
        if os.path.isdir("/opt/ml/processing/model"):
            self.model_dir = "/opt/ml/processing/model"
        else:
            self.model_dir = model_dir or "./fault_prediction/baseline/model"

    def load_metadata(self) -> pd.DataFrame:
        """Override: load and prepare device metadata. Must return df with entity_col."""
        raise NotImplementedError("Subclass must implement load_metadata()")

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Override: apply feature engineering (encode categoricals, fill NaN, etc.)."""
        raise NotImplementedError("Subclass must implement engineer_features()")

    def run(self, target_date: str | None = None):
        """Execute the full batch inference pipeline."""
        from datetime import date, timedelta

        if target_date is None:
            target_date = str(date.today() - timedelta(days=1))

        print(f"=== Batch Inference: {target_date} (lookback={self.lookback_days}d) ===\n")

        # 1. Load telemetry
        print("[1/4] Loading telemetry...")
        telemetry = load_telemetry_window(
            self.input_bucket, self.telemetry_prefix, target_date, self.lookback_days, self.region
        )
        print(f"  Loaded {len(telemetry)} rows")

        # 2. Aggregate
        print("\n[2/4] Aggregating per device...")
        features = aggregate_telemetry(telemetry, entity_col=self.entity_col)
        print(f"  {len(features)} devices")

        # 3. Join metadata
        print("\n[3/4] Joining metadata...")
        metadata = self.load_metadata()
        features = features.merge(metadata, on=self.entity_col, how="left")

        # Engineer features
        entity_ids = features[self.entity_col].copy()
        features_eng = self.engineer_features(features.drop(columns=[self.entity_col]))

        # 4. Predict
        print(f"\n[4/4] Predicting ({len(features_eng)} devices)...")
        predictions = predict_proba(self.model_dir, features_eng)
        predictions.insert(0, self.entity_col, entity_ids.values)
        predictions.insert(1, "prediction_date", target_date)

        # Upload
        import tempfile
        output_key = f"predictions/{target_date.replace('-', '')}/predictions.csv"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            predictions.to_csv(tmp, index=False)
            tmp_path = tmp.name
        try:
            boto3.client("s3", region_name=self.region).upload_file(
                tmp_path, self.output_bucket, output_key
            )
        finally:
            os.unlink(tmp_path)
        print(f"\n✅ Saved {len(predictions)} predictions to s3://{self.output_bucket}/{output_key}")
        return predictions
