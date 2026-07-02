#!/usr/bin/env python3
"""CLI for survival model training.

Usage:
    uv run python pdm/survival/train.py --train ./data/raw_train.csv --test ./data/raw_test.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from pdm.survival.model import SurvivalPredictor


def main():
    parser = argparse.ArgumentParser(description="Train survival analysis model")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("./model"))
    parser.add_argument("--time-limit", type=int, default=300)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    model = SurvivalPredictor()
    result = model.train(train_df, test_df, time_limit=args.time_limit)
    model.save(args.output)

    (args.output / "metrics.json").write_text(json.dumps(result.metrics, indent=2))
    print(f"✅ {result.metadata['model_type']} | C-index: {result.metrics['concordance_index']:.4f}")
    print(f"   Saved to {args.output}/")


if __name__ == "__main__":
    main()
