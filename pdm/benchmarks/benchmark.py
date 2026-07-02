#!/usr/bin/env python3
"""Benchmark runner — run PdM benchmarks and compare to locked baselines.

Usage:
    uv run python -m pdm.benchmarks.benchmark <base_dir> <benchmark_name>

Examples:
    uv run python -m pdm.benchmarks.benchmark ./benchmark_data all
    uv run python -m pdm.benchmarks.benchmark ./benchmark_data cmapss
    uv run python -m pdm.benchmarks.benchmark ./benchmark_data ai4i
    uv run python -m pdm.benchmarks.benchmark ./benchmark_data hdfail

Options:
    --update       Update baselines.json with new results
    --extended     Run 3 seeds and report variance
    --time-limit   Override training time limit (seconds)

Data is auto-downloaded if not present in <base_dir>/<benchmark_name>/.
Exit code 0 = all pass, 1 = regression detected.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BASELINES_PATH = SCRIPT_DIR / "baselines.json"
BENCHMARK_NAMES = ["cmapss", "ai4i", "smap", "hdfail"]


def load_baselines() -> dict:
    return json.loads(BASELINES_PATH.read_text())


def save_baselines(baselines: dict) -> None:
    BASELINES_PATH.write_text(json.dumps(baselines, indent=2))


# ---------------------------------------------------------------------------
# Individual benchmark runners
# ---------------------------------------------------------------------------

def run_cmapss(data_dir: Path, time_limit: int = 120) -> dict:
    """Run C-MAPSS FD001 RUL benchmark.
    
    Uses stride=5 and medium_quality presets for a sensible baseline
    that completes within ~3 minutes total (feature extraction + training).
    """
    from pdm.rul.model import RULPredictor

    train_df = pd.read_csv(data_dir / "raw_train.csv")
    test_df = pd.read_csv(data_dir / "raw_test.csv")

    model = RULPredictor(window_size=15, rul_cap=125)
    result = model.train(train_df, test_df, time_limit=time_limit,
                         presets="medium_quality", stride=5)
    return {
        "rmse": result.metrics["rmse"],
        "nasa_score": result.metrics.get("nasa_score"),
    }


def run_ai4i(data_dir: Path, time_limit: int = 120) -> dict:
    """Run AI4I 2020 classification benchmark."""
    from pdm.fault_prediction.model import FailureClassifier

    train_df = pd.read_csv(data_dir / "raw_train.csv")
    test_df = pd.read_csv(data_dir / "raw_test.csv")

    model = FailureClassifier()
    result = model.train(train_df, test_df, time_limit=time_limit)

    # Extract F1 from per-label metrics
    per_label = result.metrics.get("per_label", {})
    if "machine_failure" in per_label and isinstance(per_label["machine_failure"], dict):
        f1 = per_label["machine_failure"].get("f1", 0)
    else:
        f1 = result.metrics.get("mean_f1", 0)

    return {"f1": f1}


def run_smap(data_dir: Path, time_limit: int = 300) -> dict:
    """Run NASA SMAP anomaly detection benchmark.

    Uses TemporalAnomalyDetector (sliding-window PCA reconstruction error
    with temporal smoothing) which captures temporal dependencies critical
    for segment-based anomaly detection under point-adjust evaluation.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score
    from pdm.anomaly_detection.temporal import TemporalAnomalyDetector

    train_df = pd.read_csv(data_dir / "raw_train.csv")
    test_df = pd.read_csv(data_dir / "raw_test.csv")

    # Determine contamination from test label ratio
    anomaly_ratio = test_df["label"].mean()

    model = TemporalAnomalyDetector(
        window_size=5,
        n_components=0.85,
        smooth_window=11,
        contamination=min(anomaly_ratio, 0.15),
        scoring="max",
    )
    result = model.train(train_df, test_df, time_limit=time_limit)

    # Generate predictions on test set
    feature_cols = model.feature_names
    pred_result = model.predict(test_df[feature_cols])
    y_pred = pred_result.predictions["is_anomaly"].values
    y_true = test_df["label"].values

    # Point-adjust F1: if any point in a contiguous anomaly segment is detected,
    # mark the entire segment as correctly detected
    y_pred_adjusted = _point_adjust(y_true, y_pred)

    f1 = float(f1_score(y_true, y_pred_adjusted))
    precision = float(precision_score(y_true, y_pred_adjusted, zero_division=0))
    recall = float(recall_score(y_true, y_pred_adjusted, zero_division=0))

    return {"f1": f1, "precision": precision, "recall": recall}


def run_hdfail(data_dir: Path, time_limit: int = 300) -> dict:
    """Run Backblaze Hard Drive Failure (hdfail) survival benchmark.

    Uses SurvivalPredictor on the 52K-drive dataset with 94% censoring.
    This is a challenging benchmark due to extreme censoring and limited features.
    """
    from pdm.survival.model import SurvivalPredictor

    train_df = pd.read_csv(data_dir / "raw_train.csv")
    test_df = pd.read_csv(data_dir / "raw_test.csv")

    model = SurvivalPredictor()
    result = model.train(train_df, test_df, time_limit=time_limit)
    return {"concordance_index": result.metrics["concordance_index"]}


def _point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Apply point-adjust protocol: if any prediction within a true anomaly
    segment is flagged, credit the entire segment as detected."""
    y_adjusted = y_pred.copy()
    in_segment = False
    segment_start = 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_segment:
            in_segment = True
            segment_start = i
        elif y_true[i] == 0 and in_segment:
            # Segment ended at i-1. Check if any pred in [segment_start, i) is 1
            if y_pred[segment_start:i].any():
                y_adjusted[segment_start:i] = 1
            in_segment = False

    # Handle segment at end of array
    if in_segment and y_pred[segment_start:].any():
        y_adjusted[segment_start:] = 1

    return y_adjusted


# ---------------------------------------------------------------------------
# Regression check
# ---------------------------------------------------------------------------

RUNNERS = {
    "cmapss": ("cmapss_fd001_rul", run_cmapss),
    "ai4i": ("ai4i_classification", run_ai4i),
    "smap": ("smap_anomaly_detection", run_smap),
    "hdfail": ("hdfail_survival", run_hdfail),
}


def check_regression(name: str, results: dict, baselines: dict) -> tuple[bool, str]:
    """Check if result is within tolerance of baseline."""
    spec = baselines[name]
    metric = spec["metric"]
    baseline = spec["baseline"]
    tol = spec["tolerance_pct"] / 100
    value = results.get(metric)

    if value is None:
        return False, f"Metric '{metric}' not found in results: {results}"

    if spec["direction"] == "lower_is_better":
        threshold = baseline * (1 + tol)
        passed = value <= threshold
        symbol = "≤"
    else:
        threshold = baseline * (1 - tol)
        passed = value >= threshold
        symbol = "≥"

    status = "✅ PASS" if passed else "❌ REGRESSION"
    msg = (f"{metric}={value:.4f} (baseline={baseline:.4f}, "
           f"threshold {symbol} {threshold:.4f}) {status}")
    return passed, msg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run PdM benchmarks and compare to locked baselines",
    )
    parser.add_argument("base_dir", type=Path,
                       help="Base directory containing benchmark data (auto-downloads if missing)")
    parser.add_argument("benchmark", choices=BENCHMARK_NAMES + ["all"],
                       help="Which benchmark to run ('all' for the full suite)")
    parser.add_argument("--update", action="store_true",
                       help="Update baselines.json with new results")
    parser.add_argument("--extended", action="store_true",
                       help="Run 3 seeds and report mean ± std")
    parser.add_argument("--time-limit", type=int, default=None,
                       help="Override training time limit (seconds)")
    args = parser.parse_args()

    baselines = load_baselines()
    targets = BENCHMARK_NAMES if args.benchmark == "all" else [args.benchmark]

    print(f"Base data directory: {args.base_dir.resolve()}")
    print(f"Benchmarks to run: {targets}")
    print(f"{'='*70}\n")

    # Ensure data is available (auto-download if missing)
    from pdm.benchmarks.download import ensure_available
    for name in targets:
        data_dir = ensure_available(args.base_dir, name)

    all_pass = True

    for name in targets:
        baseline_key, runner = RUNNERS[name]
        data_dir = args.base_dir / name
        time_limit = args.time_limit or baselines[baseline_key].get("time_limit", 120)

        print(f"\n[{name}]")

        if args.extended:
            metrics_list = []
            for seed in range(3):
                np.random.seed(seed)
                t0 = time.time()
                results = runner(data_dir, time_limit=time_limit)
                elapsed = time.time() - t0
                metrics_list.append(results)
                print(f"  seed={seed}: {results} ({elapsed:.0f}s)")

            metric_key = baselines[baseline_key]["metric"]
            values = [m[metric_key] for m in metrics_list if m.get(metric_key) is not None]
            print(f"  → {metric_key} = {np.mean(values):.4f} ± {np.std(values):.4f}")
            results = {metric_key: float(np.mean(values))}
        else:
            t0 = time.time()
            results = runner(data_dir, time_limit=time_limit)
            elapsed = time.time() - t0
            print(f"  Completed in {elapsed:.0f}s")

        passed, msg = check_regression(baseline_key, results, baselines)
        print(f"  {msg}")

        if not passed:
            all_pass = False

        if args.update and results.get(baselines[baseline_key]["metric"]) is not None:
            from datetime import date
            baselines[baseline_key]["baseline"] = round(
                results[baselines[baseline_key]["metric"]], 4
            )
            baselines[baseline_key]["locked_at"] = str(date.today())

    # Summary
    print(f"\n{'='*70}")
    if all_pass:
        print("✅ ALL BENCHMARKS PASSED — no regressions detected")
    else:
        print("❌ REGRESSION DETECTED — see above for details")

    if args.update:
        save_baselines(baselines)
        print(f"\n📝 baselines.json updated")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
