#!/usr/bin/env python3
"""Inject synthetic anomalies into test data for unsupervised AD validation.

Three anomaly types:
  - spike: sudden value jump (point anomaly)
  - drift: gradual increase over consecutive samples (collective anomaly)
  - level_shift: permanent offset change (contextual anomaly)

Usage:
    from pdm.anomaly_detection.synthetic_anomalies import inject_anomalies
    augmented_df, injection_labels = inject_anomalies(test_df, feature_cols, fraction=0.1)
"""
import numpy as np
import pandas as pd


def inject_anomalies(
    df: pd.DataFrame,
    feature_cols: list[str],
    fraction: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Inject synthetic anomalies into a copy of df.

    Returns:
        (augmented_df, labels) where labels is 1 for injected anomaly rows, 0 otherwise.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    # Cast feature columns to float to avoid dtype incompatibility warnings
    for col in feature_cols:
        if col in out.columns:
            out[col] = out[col].astype(float)
    n = len(df)
    n_anomalies = max(1, int(n * fraction))

    # Select rows to corrupt
    anomaly_indices = rng.choice(n, size=n_anomalies, replace=False)
    labels = pd.Series(0, index=df.index)
    labels.iloc[anomaly_indices] = 1

    # Split into 3 roughly equal groups
    splits = np.array_split(anomaly_indices, 3)

    for idx in splits[0]:  # spike
        col = rng.choice(feature_cols)
        std = df[col].std()
        if std > 0:
            out.at[out.index[idx], col] += rng.choice([-1, 1]) * rng.uniform(4, 8) * std

    for idx in splits[1]:  # drift (corrupt this + next 2 rows if available)
        col = rng.choice(feature_cols)
        std = df[col].std()
        if std > 0:
            for offset in range(min(3, n - idx)):
                out.at[out.index[idx + offset], col] += (offset + 1) * rng.uniform(1, 3) * std
                if idx + offset not in anomaly_indices and offset > 0:
                    labels.iloc[idx + offset] = 1

    for idx in splits[2]:  # level_shift
        col = rng.choice(feature_cols)
        std = df[col].std()
        if std > 0:
            out.at[out.index[idx], col] += rng.choice([-1, 1]) * rng.uniform(3, 5) * std

    return out, labels
