"""PyTorch Dataset for windowed multivariate time series.

Each sample is a (n_sensors, window_size) tensor with a scalar target.
"""
import numpy as np
import pandas as pd


class WindowedTSDataset:
    """Sliding window dataset for RUL or classification.

    Extracts windows of shape (window_size, n_sensors) from each unit's time series,
    paired with the target value at the window's end.

    Args:
        df: DataFrame with entity, sensor, and target columns
        sensor_cols: Columns containing sensor readings
        target_col: Column with the target (RUL, label, etc.)
        entity_col: Column identifying each unit/device
        window_size: Number of timesteps per window
    """

    def __init__(self, df: pd.DataFrame, sensor_cols: list[str],
                 target_col: str, entity_col: str, window_size: int = 30):
        self.samples: list[np.ndarray] = []
        self.targets: list[float] = []

        for _, group in df.groupby(entity_col):
            values = group[sensor_cols].values.astype(np.float32)
            targets = group[target_col].values.astype(np.float32)

            for i in range(window_size - 1, len(values)):
                window = values[i - window_size + 1: i + 1]  # (W, C)
                self.samples.append(window)
                self.targets.append(float(targets[i]))

        self.samples = np.array(self.samples)  # (N, W, C)
        self.targets = np.array(self.targets)  # (N,)

        # Normalize per-channel
        self.mean = self.samples.mean(axis=(0, 1))  # (C,)
        self.std = self.samples.std(axis=(0, 1)) + 1e-8
        self.samples = (self.samples - self.mean) / self.std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Returns (x, y) where x is (C, W) channels-first for tsai/PyTorch conv layers."""
        try:
            import torch
            x = torch.tensor(self.samples[idx].T, dtype=torch.float32)  # (C, W)
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, y
        except ImportError:
            # Return numpy arrays if torch not available
            return self.samples[idx].T, self.targets[idx]

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Return all data as numpy arrays in channels-first format (N, C, W)."""
        X = self.samples.transpose(0, 2, 1)  # (N, C, W)
        return X, self.targets
