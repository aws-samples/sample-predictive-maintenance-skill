"""Foundation model embeddings for zero-shot anomaly detection.

Uses pretrained time series foundation models (Chronos) to extract
embeddings, then applies outlier detection in embedding space.
No task-specific training needed — works on new equipment with no history.

Requires: pip install chronos-forecasting torch

Usage:
    from pdm.anomaly_detection.foundation import FoundationAnomalyDetector

    # Zero-shot (no reference data needed)
    detector = FoundationAnomalyDetector(model_size="small")
    results = detector.predict_zero_shot(df, sensor_cols, entity_col="unit_id")

    # With reference (fit on healthy data, detect deviations)
    detector.fit(healthy_df, sensor_cols, entity_col="unit_id")
    results = detector.predict(new_df)
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class ChronosEmbedder:
    """Extract time series embeddings using Amazon Chronos.

    Chronos is a T5-family model pretrained on billions of time series observations.
    We use its encoder to produce fixed-size representations of sensor windows.
    """

    def __init__(self, model_size: str = "small", device: str = "auto"):
        """
        Args:
            model_size: "tiny" (8M), "mini" (20M), "small" (46M), "base" (200M), "large" (710M)
            device: "auto", "cpu", or "cuda"
        """
        self.model_size = model_size
        self.device = device
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return
        try:
            from chronos import ChronosPipeline
            import torch
            device = self.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._pipeline = ChronosPipeline.from_pretrained(
                f"amazon/chronos-t5-{self.model_size}",
                device_map=device,
                torch_dtype=torch.float32,
            )
        except ImportError:
            raise ImportError("Foundation models require: pip install chronos-forecasting torch")

    def embed(self, series: np.ndarray) -> np.ndarray:
        """Embed a single time series (1D array) into a fixed-size vector.

        Args:
            series: 1D array of shape (seq_len,)

        Returns:
            1D array of shape (d_model,) — mean-pooled encoder output
        """
        self._load()
        import torch
        tensor = torch.tensor(series, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            embedding = self._pipeline.embed(tensor)  # (1, seq_len, d_model)
            pooled = embedding.mean(dim=1).squeeze().cpu().numpy()
        return pooled

    def embed_windows(
        self,
        df: pd.DataFrame,
        sensor_cols: list[str],
        entity_col: str,
        window_size: int = 64,
        stride: int = 32,
    ) -> np.ndarray:
        """Extract embeddings for each window across all sensors.

        Returns:
            Array of shape (n_windows, d_model * n_sensors)
        """
        self._load()
        import torch

        all_embeddings = []
        for _, group in df.groupby(entity_col):
            values = group[sensor_cols].values
            for i in range(window_size - 1, len(values), stride):
                window_embs = []
                for j, col in enumerate(sensor_cols):
                    series = values[i - window_size + 1: i + 1, j].astype(np.float32)
                    emb = self.embed(series)
                    window_embs.append(emb)
                all_embeddings.append(np.concatenate(window_embs))

        return np.array(all_embeddings) if all_embeddings else np.empty((0, 0))


class FoundationAnomalyDetector:
    """Zero-shot and few-shot anomaly detection using foundation model embeddings.

    Two modes:
    1. fit() + predict(): Learn normal embedding space, flag deviations
    2. predict_zero_shot(): No training needed — uses LOF in embedding space
    """

    def __init__(self, model_size: str = "small", contamination: float = 0.05,
                 window_size: int = 64):
        self.embedder = ChronosEmbedder(model_size=model_size)
        self.contamination = contamination
        self.window_size = window_size
        self.centroid: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None
        self.sensor_cols: list[str] = []
        self.entity_col: str = ""

    def fit(self, normal_df: pd.DataFrame, sensor_cols: list[str],
            entity_col: str) -> "FoundationAnomalyDetector":
        """Build reference embedding space from healthy/normal data."""
        self.sensor_cols = sensor_cols
        self.entity_col = entity_col

        embeddings = self.embedder.embed_windows(
            normal_df, sensor_cols, entity_col, window_size=self.window_size,
        )

        self.centroid = embeddings.mean(axis=0)
        distances = np.linalg.norm(embeddings - self.centroid, axis=1)
        self.threshold = float(np.percentile(distances, (1 - self.contamination) * 100))
        return self

    def predict(self, new_df: pd.DataFrame) -> pd.DataFrame:
        """Score new data against the learned reference space."""
        if self.centroid is None:
            raise RuntimeError("Must call fit() first, or use predict_zero_shot()")

        embeddings = self.embedder.embed_windows(
            new_df, self.sensor_cols, self.entity_col, window_size=self.window_size,
        )
        distances = np.linalg.norm(embeddings - self.centroid, axis=1)
        return pd.DataFrame({
            "embedding_distance": distances,
            "is_anomaly": (distances > self.threshold).astype(int),
        })

    def predict_zero_shot(self, df: pd.DataFrame, sensor_cols: list[str],
                          entity_col: str) -> pd.DataFrame:
        """Zero-shot anomaly detection — no training required.

        Uses Local Outlier Factor in embedding space. Anomalies are points
        that are outliers among the embeddings themselves.
        """
        from sklearn.neighbors import LocalOutlierFactor

        embeddings = self.embedder.embed_windows(
            df, sensor_cols, entity_col, window_size=self.window_size,
        )

        if len(embeddings) < 5:
            return pd.DataFrame({"anomaly_score": [], "is_anomaly": []})

        n_neighbors = min(20, len(embeddings) - 1)
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.contamination,
            novelty=False,
        )
        labels = lof.fit_predict(embeddings)
        scores = -lof.negative_outlier_factor_

        return pd.DataFrame({
            "anomaly_score": scores,
            "is_anomaly": (labels == -1).astype(int),
        })
