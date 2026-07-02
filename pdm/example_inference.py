"""Example inference: score one day of telemetry with trained model + optional explanations.

Usage:
    .venv/bin/python pdm/example_inference.py \
        --bucket SOURCE_BUCKET \
        --date 2026-04-26 \
        --model-dir ./fault_prediction/baseline/model \
        --runtime ./fault_prediction/baseline/runtime.py \
        --output ./data/example_predictions.csv \
        --explain
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

from pdm.data.data_exploration import (
    load_all_flat_parquet,
    load_partitioned_parquet,
)
from pdm.data.utils import (
    align_to_model,
    deduplicate_on,
    booleans_to_int,
    safe_age_days,
    feature_contributions,
    load_eav_chunked,
    pivot_precomputed_eav_temporal,
)


def load_runtime(runtime_path: str):
    """Dynamically import a runtime.py module and return its engineer_features function."""
    spec = importlib.util.spec_from_file_location("runtime", runtime_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.engineer_features


def load_telemetry_for_date(bucket: str, prefix: str, date: str) -> pd.DataFrame:
    """Load one day of telemetry from a partitioned EAV table."""
    df = load_partitioned_parquet(
        bucket, prefix,
        columns=["device_id", "sensor_name", "sample_value"],
        max_partitions=1,
        partition_filter=date,
    )
    return df


def aggregate_telemetry_wide(telem: pd.DataFrame, entity_col: str = "device_id") -> pd.DataFrame:
    """Aggregate EAV telemetry to wide format (one row per device)."""
    stats = telem.groupby([entity_col, "sensor_name"])["sample_value"].agg(
        ["mean", "std", "max", "min"]
    ).reset_index()
    pivot = stats.pivot_table(
        index=entity_col, columns="sensor_name", values=["mean", "std", "max", "min"]
    )
    pivot.columns = [f"{sensor}_{stat}" for stat, sensor in pivot.columns]
    return pivot.reset_index()


def run_example_inference(
    bucket: str,
    telemetry_prefix: str,
    inference_date: str,
    model_dir: str,
    runtime_path: str,
    output_path: str,
    metadata_prefix: str | None = None,
    metadata_bucket: str | None = None,
    entity_col: str = "device_id",
    explain: bool = False,
    explain_top_k: int = 5,
) -> pd.DataFrame:
    """Run inference on a single day of telemetry.

    Args:
        bucket: S3 bucket containing telemetry data.
        telemetry_prefix: S3 prefix for telemetry (e.g., 'telemetry/').
        inference_date: Date string to score (e.g., '2026-04-26').
        model_dir: Path to model directory (containing metadata.json + ag_model/).
        runtime_path: Path to the experiment's runtime.py with engineer_features().
        output_path: Where to save predictions CSV.
        metadata_prefix: S3 prefix for device metadata (optional).
        metadata_bucket: S3 bucket for metadata (defaults to telemetry bucket).
        entity_col: Device ID column name.
        explain: If True, compute perturbation-based feature explanations.
        explain_top_k: Number of top features per explanation.

    Returns:
        DataFrame with predictions and optional explanations.
    """
    from autogluon.tabular import TabularPredictor

    # Load model metadata
    metadata = json.load(open(os.path.join(model_dir, "metadata.json")))
    model_features = metadata["feature_names"]
    label_names = metadata["label_names"]
    formulation = metadata.get("formulation", "multilabel")

    # Load + aggregate telemetry
    print(f"Loading telemetry for {inference_date}...")
    telem = load_telemetry_for_date(bucket, telemetry_prefix, inference_date)
    print(f"  {len(telem)} rows, {telem[entity_col].nunique()} devices")
    wide = aggregate_telemetry_wide(telem, entity_col)
    print(f"  Aggregated: {wide.shape}")

    # Load metadata if available
    if metadata_prefix:
        meta_bucket = metadata_bucket or bucket
        meta = load_all_flat_parquet(meta_bucket, metadata_prefix)
        meta[entity_col] = meta[entity_col].astype(str)
        meta = deduplicate_on(meta, entity_col)
        bool_cols = [c for c in meta.columns if meta[c].dtype == "bool"]
        if bool_cols:
            meta = booleans_to_int(meta, bool_cols)
        # Convert date columns to age
        date_cols = [c for c in meta.columns if "date" in c.lower() and meta[c].dtype.name.startswith("date")]
        for col in date_cols:
            age_col = col.replace("_date", "").replace("_at", "") + "_age_days"
            meta[age_col] = safe_age_days(meta[col])
            meta = meta.drop(columns=[col])
        wide[entity_col] = wide[entity_col].astype(str)
        wide = wide.merge(meta, on=entity_col, how="left")

    # Engineer features
    device_ids = wide[entity_col].copy()
    wide["_observation_date"] = inference_date
    engineer_features = load_runtime(runtime_path)
    features = engineer_features(wide)

    # Align to model
    features = align_to_model(features, os.path.join(model_dir, "metadata.json"))
    print(f"  Features aligned: {features.shape}")

    # Predict
    results = pd.DataFrame({entity_col: device_ids, "observation_date": inference_date})
    predictors = {}
    for label in label_names:
        ag_path = os.path.join(model_dir, "ag_model", label)
        p = TabularPredictor.load(ag_path, verbosity=0)
        predictors[label] = p
        proba = p.predict_proba(features)[1].values
        results[f"{label}_probability"] = np.round(proba, 4)
        results[f"{label}_prediction"] = (proba >= 0.5).astype(int)

    # Explanations
    if explain:
        print("Computing feature explanations for positive predictions...")
        metrics_path = os.path.join(model_dir, "metrics.json")
        top_features_per_label = {}
        if os.path.exists(metrics_path):
            metrics = json.load(open(metrics_path))
            fi = metrics.get("feature_importance", {})
            for label in label_names:
                top_features_per_label[label] = [
                    f["feature"] for f in fi.get(label, [])[:10]
                ]

        all_explanations = []
        for i in range(len(features)):
            row_expl = []
            for label in label_names:
                if results.iloc[i][f"{label}_prediction"] == 1:
                    predictor = predictors[label]
                    predict_fn = lambda x, p=predictor: p.predict_proba(x)[1].values[0]
                    feats_to_check = top_features_per_label.get(label, model_features[:20])
                    row = features.iloc[i : i + 1].copy()
                    contribs = feature_contributions(predict_fn, row, feats_to_check, explain_top_k)
                    expl_str = "; ".join([f"{k}(Δ={v:+.4f})" for k, v in contribs])
                    row_expl.append(f"{label}: {expl_str}")
            all_explanations.append(" | ".join(row_expl) if row_expl else "")
        results["top_features"] = all_explanations

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    results.to_csv(output_path, index=False)

    # Summary
    print(f"\n=== Example Inference Results ===")
    print(f"Devices scored: {len(results)}")
    n_total_positive = 0
    for label in label_names:
        n_pos = results[f"{label}_prediction"].sum()
        n_total_positive += n_pos
        if n_pos > 0:
            print(f"  {label}: {n_pos} predicted positive")
    if n_total_positive == 0:
        print("  No positive predictions at threshold=0.5")
    print(f"\nSaved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run example inference on one day of telemetry")
    parser.add_argument("--bucket", required=True, help="S3 bucket with telemetry")
    parser.add_argument("--telemetry-prefix", default="telemetry/", help="S3 prefix for telemetry")
    parser.add_argument("--date", required=True, help="Inference date (YYYY-MM-DD)")
    parser.add_argument("--model-dir", required=True, help="Path to model directory")
    parser.add_argument("--runtime", required=True, help="Path to runtime.py")
    parser.add_argument("--output", default="./data/example_predictions.csv")
    parser.add_argument("--metadata-prefix", default=None, help="S3 prefix for device metadata")
    parser.add_argument("--metadata-bucket", default=None, help="S3 bucket for metadata (default: same as --bucket)")
    parser.add_argument("--entity-col", default="device_id", help="Device ID column name")
    parser.add_argument("--explain", action="store_true", help="Compute feature explanations")
    parser.add_argument("--explain-top-k", type=int, default=5)
    args = parser.parse_args()

    run_example_inference(
        bucket=args.bucket,
        telemetry_prefix=args.telemetry_prefix,
        inference_date=args.date,
        model_dir=args.model_dir,
        runtime_path=args.runtime,
        output_path=args.output,
        metadata_prefix=args.metadata_prefix,
        metadata_bucket=args.metadata_bucket,
        entity_col=args.entity_col,
        explain=args.explain,
        explain_top_k=args.explain_top_k,
    )


if __name__ == "__main__":
    main()
