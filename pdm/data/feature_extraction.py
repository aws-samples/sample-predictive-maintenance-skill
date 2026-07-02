"""Time series feature extraction — tsfresh wrapper with manual fallback.

Provides automated feature extraction from windowed sensor data. Uses tsfresh
when available; falls back to the manual sliding_window_features() otherwise.

Usage:
    from pdm.data.feature_extraction import extract_features

    features = extract_features(
        df, entity_col="unit_id", time_col="cycle",
        sensor_cols=["s1", "s2"], window_size=30,
        target=train_df["RUL"],  # enables relevance filtering
    )
"""
from typing import Optional

import numpy as np
import pandas as pd


def extract_features(
    df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    sensor_cols: list[str],
    window_size: int = 30,
    stride: int = 1,
    target: Optional[pd.Series] = None,
    mode: str = "auto",
    settings: str = "efficient",
    n_jobs: int = 0,
) -> pd.DataFrame:
    """Extract features from windowed time series data.

    Args:
        df: Raw data with entity, time, and sensor columns
        entity_col: Column identifying each unit/device
        time_col: Column with time ordering (cycle, timestamp)
        sensor_cols: Columns containing sensor readings
        window_size: Window size for rolling extraction
        stride: Stride between consecutive windows
        target: If provided with tsfresh, enables relevance filtering
        mode: "auto" (try tsfresh, fallback to manual), "tsfresh", "manual"
        settings: tsfresh preset: "minimal" (~30/sensor), "efficient" (~200), "comprehensive" (~794)
        n_jobs: Parallel workers (0 = auto for tsfresh)

    Returns:
        DataFrame with one row per window and extracted features
    """
    if mode == "manual":
        return _extract_manual(df, entity_col, sensor_cols, window_size, stride)

    if mode == "auto":
        try:
            import tsfresh  # noqa: F401
            return _extract_tsfresh(df, entity_col, time_col, sensor_cols,
                                    window_size, stride, target, settings, n_jobs)
        except ImportError:
            return _extract_manual(df, entity_col, sensor_cols, window_size, stride)

    if mode == "tsfresh":
        return _extract_tsfresh(df, entity_col, time_col, sensor_cols,
                                window_size, stride, target, settings, n_jobs)

    raise ValueError(f"Unknown mode: {mode}. Use 'auto', 'tsfresh', or 'manual'.")


def _extract_tsfresh(
    df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    sensor_cols: list[str],
    window_size: int,
    stride: int,
    target: Optional[pd.Series],
    settings: str,
    n_jobs: int,
) -> pd.DataFrame:
    """Extract features using tsfresh."""
    from tsfresh import extract_features as tf_extract, select_features
    from tsfresh.feature_extraction import (
        MinimalFCParameters, EfficientFCParameters, ComprehensiveFCParameters,
    )
    from tsfresh.utilities.dataframe_functions import roll_time_series

    settings_map = {
        "minimal": MinimalFCParameters(),
        "efficient": EfficientFCParameters(),
        "comprehensive": ComprehensiveFCParameters(),
    }
    fc_params = settings_map.get(settings, EfficientFCParameters())

    # Prepare tsfresh format: [id, sort, value_cols...]
    ts_df = df[[entity_col, time_col] + sensor_cols].copy()
    ts_df = ts_df.rename(columns={entity_col: "id", time_col: "sort"})

    # Roll into windows
    rolled = roll_time_series(
        ts_df, column_id="id", column_sort="sort",
        max_timeshift=window_size - 1, min_timeshift=window_size - 1,
    )

    # Extract
    features = tf_extract(
        rolled, column_id="id", column_sort="sort",
        default_fc_parameters=fc_params,
        n_jobs=n_jobs, disable_progressbar=True,
    )

    # Clean
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna(axis=1, how="any")
    features = features.loc[:, features.std() > 0]

    # Relevance filtering
    if target is not None:
        aligned_target = target.reindex(features.index).dropna()
        common_idx = features.index.intersection(aligned_target.index)
        if len(common_idx) > 10:
            features = select_features(features.loc[common_idx], aligned_target.loc[common_idx])

    return features


def _extract_manual(
    df: pd.DataFrame,
    entity_col: str,
    sensor_cols: list[str],
    window_size: int,
    stride: int,
) -> pd.DataFrame:
    """Fallback: use our manual sliding_window_features."""
    from pdm.rul.model import sliding_window_features

    features, indices = sliding_window_features(
        df, unit_col=entity_col, sensor_cols=sensor_cols,
        window_size=window_size, stride=stride,
    )
    features.index = indices
    return features
