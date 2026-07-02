"""Uncertainty quantification via conformal prediction.

Provides distribution-free prediction intervals with guaranteed coverage.
No external dependencies beyond numpy.

Usage:
    from pdm.uncertainty import ConformalPredictor

    # After training a model:
    conformal = ConformalPredictor(confidence=0.9)
    conformal.calibrate(y_true_cal, y_pred_cal)

    # At inference:
    intervals = conformal.predict_interval(y_pred_new)
    # intervals has: prediction, lower_bound, upper_bound, interval_width
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


class ConformalPredictor:
    """Split conformal prediction for regression models.

    Produces prediction intervals with guaranteed marginal coverage:
    P(Y ∈ [lower, upper]) ≥ confidence.

    Requires a calibration set (held-out data not used during training).
    """

    def __init__(self, confidence: float = 0.9):
        """
        Args:
            confidence: Desired coverage probability (e.g., 0.9 = 90% of true
                       values will fall within the predicted interval)
        """
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between 0 and 1")
        self.confidence = confidence
        self.quantile: float | None = None
        self.n_calibration: int = 0

    @property
    def is_calibrated(self) -> bool:
        return self.quantile is not None

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> "ConformalPredictor":
        """Calibrate using residuals from a held-out calibration set.

        Args:
            y_true: True values from calibration set
            y_pred: Model predictions on calibration set

        Returns:
            self (for chaining)
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
        if len(y_true) < 10:
            raise ValueError(f"Need at least 10 calibration samples, got {len(y_true)}")

        residuals = np.abs(y_true - y_pred)
        n = len(residuals)

        # Conformal quantile with finite-sample correction
        q_level = np.ceil((n + 1) * self.confidence) / n
        self.quantile = float(np.quantile(residuals, min(q_level, 1.0)))
        self.n_calibration = n
        return self

    def predict_interval(self, y_pred: np.ndarray) -> pd.DataFrame:
        """Generate prediction intervals around point predictions.

        Args:
            y_pred: Point predictions from the model

        Returns:
            DataFrame with columns: prediction, lower_bound, upper_bound,
            interval_width, confidence
        """
        if not self.is_calibrated:
            raise RuntimeError("Must call calibrate() before predict_interval()")

        y_pred = np.asarray(y_pred).flatten()
        return pd.DataFrame({
            "prediction": y_pred,
            "lower_bound": y_pred - self.quantile,
            "upper_bound": y_pred + self.quantile,
            "interval_width": np.full(len(y_pred), self.quantile * 2),
            "confidence": np.full(len(y_pred), self.confidence),
        })

    def coverage(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute empirical coverage on a test set.

        Returns fraction of true values within predicted intervals.
        Should be ≥ self.confidence if the calibration set is exchangeable.
        """
        if not self.is_calibrated:
            raise RuntimeError("Must call calibrate() first")
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        lower = y_pred - self.quantile
        upper = y_pred + self.quantile
        return float(((y_true >= lower) & (y_true <= upper)).mean())

    def save(self, path: Path) -> None:
        """Save calibration state to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "confidence": self.confidence,
            "quantile": self.quantile,
            "n_calibration": self.n_calibration,
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> "ConformalPredictor":
        """Load calibrated predictor from JSON."""
        data = json.loads(Path(path).read_text())
        obj = cls(confidence=data["confidence"])
        obj.quantile = data["quantile"]
        obj.n_calibration = data["n_calibration"]
        return obj
