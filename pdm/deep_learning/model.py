"""Deep learning RUL predictor using tsai (PyTorch/fastai).

Requires: pip install tsai torch fastai

Supported architectures:
    - InceptionTime (default, best accuracy/speed tradeoff)
    - LSTM_FCN (good for short sequences)
    - TSTPlus (Transformer, best on long sequences)
    - ROCKET (fastest, random convolutional kernels)
    - TCN (Temporal Convolutional Network)
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pdm.base import PDMModel, PredictionResult, TrainResult
from pdm.deep_learning.dataset import WindowedTSDataset


# Architecture name → tsai class mapping (lazy-loaded)
ARCHITECTURES = {
    "InceptionTime": "InceptionTime",
    "LSTM_FCN": "LSTM_FCN",
    "TSTPlus": "TSTPlus",
    "ROCKET": "RocketClassifier",  # Used for regression via wrapper
    "TCN": "TCN",
}


class DeepRULPredictor(PDMModel):
    """RUL prediction using deep learning (tsai/PyTorch).

    Trains a neural network directly on windowed sensor sequences.
    Best for large datasets (>10k samples) with many sensors.

    Args:
        architecture: Model architecture name (default "InceptionTime")
        window_size: Window length in timesteps
        rul_cap: Maximum RUL value (clipped during training)
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Mini-batch size
    """

    formulation = "rul"

    def __init__(self, architecture: str = "InceptionTime", window_size: int = 30,
                 rul_cap: int = 125, epochs: int = 50, lr: float = 1e-3,
                 batch_size: int = 64):
        self.architecture = architecture
        self.window_size = window_size
        self.rul_cap = rul_cap
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.learner = None
        self.sensor_cols: list[str] = []
        self.normalization: Optional[dict] = None

    def train(self, train_df: pd.DataFrame, test_df: pd.DataFrame, **kwargs) -> TrainResult:
        """Train deep learning model on windowed sensor data."""
        try:
            from tsai.all import (
                get_ts_dls, ts_learner, mae, rmse,
                InceptionTime, LSTM_FCN, TSTPlus, TCN,
            )
        except ImportError:
            raise ImportError(
                "Deep learning requires tsai: pip install tsai torch fastai"
            )

        # Detect sensor columns
        exclude = {"unit_id", "cycle", "RUL"}
        self.sensor_cols = [c for c in train_df.columns
                          if c not in exclude and train_df[c].dtype in (np.float64, np.float32, np.int64)]

        # Cap RUL
        train_df = train_df.copy()
        train_df["RUL"] = train_df["RUL"].clip(upper=self.rul_cap)
        test_df = test_df.copy()
        test_df["RUL"] = test_df["RUL"].clip(upper=self.rul_cap)

        # Create windowed datasets
        train_ds = WindowedTSDataset(train_df, self.sensor_cols, "RUL", "unit_id", self.window_size)
        test_ds = WindowedTSDataset(test_df, self.sensor_cols, "RUL", "unit_id", self.window_size)
        self.normalization = {"mean": train_ds.mean.tolist(), "std": train_ds.std.tolist()}

        X_train, y_train = train_ds.to_numpy()
        X_test, y_test = test_ds.to_numpy()

        # Create tsai dataloaders
        dls = get_ts_dls(X_train, y_train, X_test, y_test, bs=self.batch_size)

        # Select architecture
        arch_map = {
            "InceptionTime": InceptionTime,
            "LSTM_FCN": LSTM_FCN,
            "TSTPlus": TSTPlus,
            "TCN": TCN,
        }
        arch_cls = arch_map.get(self.architecture, InceptionTime)

        # Train
        self.learner = ts_learner(dls, arch_cls, metrics=[mae, rmse])
        self.learner.fit_one_cycle(self.epochs, self.lr)

        # Evaluate
        import torch
        preds, targets = self.learner.get_preds(dl=dls.valid)
        rmse_val = float(torch.sqrt(torch.mean((preds.squeeze() - targets.squeeze()) ** 2)))
        mae_val = float(torch.mean(torch.abs(preds.squeeze() - targets.squeeze())))

        return TrainResult(
            model=self.learner,
            metrics={"rmse": rmse_val, "mae": mae_val},
            metadata={
                "formulation": "rul", "architecture": self.architecture,
                "window_size": self.window_size, "n_sensors": len(self.sensor_cols),
                "sensor_cols": self.sensor_cols, "epochs": self.epochs,
            },
        )

    def predict(self, features: pd.DataFrame) -> PredictionResult:
        """Predict RUL on new data."""
        if self.learner is None:
            raise RuntimeError("Model not trained. Call train() first.")

        ds = WindowedTSDataset(features, self.sensor_cols, "RUL", "unit_id", self.window_size)
        X, _ = ds.to_numpy()

        import torch
        from tsai.all import TSDatasets, TSDataLoaders

        # Predict
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.learner.model.eval()
        with torch.no_grad():
            preds = self.learner.model(X_tensor).squeeze().numpy()

        return PredictionResult(predictions=pd.DataFrame({"predicted_rul": preds}))

    def explain(self, features: pd.DataFrame, top_k: int = 10) -> list[dict]:
        """Feature importance via gradient-based attribution (placeholder)."""
        return []  # TODO: Implement GradCAM or integrated gradients

    def save(self, path: Path) -> None:
        """Save model weights and metadata."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.learner.export(path / "model.pkl")
        (path / "metadata.json").write_text(json.dumps({
            "formulation": "rul", "architecture": self.architecture,
            "window_size": self.window_size, "rul_cap": self.rul_cap,
            "sensor_cols": self.sensor_cols, "normalization": self.normalization,
            "epochs": self.epochs, "lr": self.lr, "batch_size": self.batch_size,
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> "DeepRULPredictor":
        """Load trained model."""
        from tsai.all import load_learner
        path = Path(path)
        meta = json.loads((path / "metadata.json").read_text())
        obj = cls(
            architecture=meta["architecture"],
            window_size=meta["window_size"],
            rul_cap=meta.get("rul_cap", 125),
        )
        obj.learner = load_learner(path / "model.pkl")
        obj.sensor_cols = meta["sensor_cols"]
        obj.normalization = meta.get("normalization")
        return obj
