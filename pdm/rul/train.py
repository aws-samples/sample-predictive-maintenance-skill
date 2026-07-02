#!/usr/bin/env python3
"""CLI for RUL model training.

Usage:
    uv run python pdm/rul/train.py --train ./data/raw_train.csv --test ./data/raw_test.csv
    uv run python pdm/rul/train.py --train ... --test ... --auto-window
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pdm.rul.model import RULPredictor


def search_window_size(train_df, test_df, rul_cap, time_per_window=120):
    """Try multiple window sizes with a quick model and return the best."""
    import warnings
    warnings.filterwarnings("ignore")

    # Subsample: use 30 random units for speed
    units = train_df["unit_id"].unique()
    if len(units) > 30:
        sample_units = np.random.choice(units, 30, replace=False)
        train_sub = train_df[train_df["unit_id"].isin(sample_units)]
    else:
        train_sub = train_df

    candidates = [10, 15, 20, 25, 30]
    best_rmse, best_ws = float("inf"), 30
    print("🔍 Auto window search:")
    for ws in candidates:
        with tempfile.TemporaryDirectory() as tmp:
            model = RULPredictor(window_size=ws, rul_cap=rul_cap)
            result = model.train(train_sub, test_df, time_limit=time_per_window,
                                 presets="medium_quality", output=Path(tmp))
            rmse = result.metrics["rmse"]
            print(f"   window={ws:2d} → RMSE={rmse:.2f}")
            if rmse < best_rmse:
                best_rmse, best_ws = rmse, ws
    print(f"   ✅ Best window: {best_ws} (RMSE={best_rmse:.2f})")
    return best_ws


def main():
    parser = argparse.ArgumentParser(description="Train RUL prediction model")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("./model"))
    parser.add_argument("--time-limit", type=int, default=600)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--rul-cap", type=int, default=125)
    parser.add_argument("--stride", type=int, default=1, help="Window stride (default: 1)")
    parser.add_argument("--presets", default="best")
    parser.add_argument("--auto-window", action="store_true",
                        help="Search over window sizes before final training")
    parser.add_argument("--optuna", action="store_true",
                        help="Use Optuna HPO + ensemble instead of AutoGluon")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    if args.auto_window:
        args.window_size = search_window_size(train_df, test_df, args.rul_cap)

    model = RULPredictor(window_size=args.window_size, rul_cap=args.rul_cap)
    backend = "optuna" if args.optuna else "autogluon"
    result = model.train(train_df, test_df, time_limit=args.time_limit, presets=args.presets,
                         output=args.output, backend=backend, n_trials=args.n_trials,
                         stride=args.stride)
    model.save(args.output)

    (args.output / "metrics.json").write_text(json.dumps(result.metrics, indent=2))
    print(f"✅ RMSE: {result.metrics['rmse']:.2f} | NASA: {result.metrics['nasa_score']:.1f}")
    print(f"   Saved to {args.output}/")


if __name__ == "__main__":
    main()
