"""Capture and persist reference distributions from training data.

Call capture_baseline() after training and save alongside model artifacts.
At inference time, load and compare against new data via drift.py.
"""
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class FeatureBaseline:
    """Reference distribution statistics for a single feature."""
    name: str
    mean: float
    std: float
    min: float
    max: float
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float
    n_samples: int


def capture_baseline(df: pd.DataFrame, feature_cols: list[str]) -> list[FeatureBaseline]:
    """Compute reference statistics from training data.

    Args:
        df: Training DataFrame (after feature engineering)
        feature_cols: Numeric feature columns to track

    Returns:
        List of FeatureBaseline objects, one per feature
    """
    baselines = []
    for col in feature_cols:
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        baselines.append(FeatureBaseline(
            name=col,
            mean=float(vals.mean()),
            std=float(vals.std()),
            min=float(vals.min()),
            max=float(vals.max()),
            p5=float(vals.quantile(0.05)),
            p25=float(vals.quantile(0.25)),
            p50=float(vals.quantile(0.50)),
            p75=float(vals.quantile(0.75)),
            p95=float(vals.quantile(0.95)),
            n_samples=len(vals),
        ))
    return baselines


def save_baseline(baselines: list[FeatureBaseline], path: Path) -> None:
    """Save baselines as JSON alongside model artifacts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(b) for b in baselines]
    path.write_text(json.dumps(data, indent=2))


def load_baseline(path: Path) -> list[FeatureBaseline]:
    """Load saved baselines from JSON."""
    data = json.loads(Path(path).read_text())
    return [FeatureBaseline(**d) for d in data]
