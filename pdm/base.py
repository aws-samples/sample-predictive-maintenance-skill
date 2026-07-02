"""PDM Model base classes — unified interface for all predictive maintenance models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class TrainResult:
    """Result of model training."""
    model: Any
    metrics: dict[str, float]
    feature_importance: Optional[pd.Series] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Unified prediction output."""
    predictions: pd.DataFrame
    explanations: Optional[list[dict]] = None


class PDMModel(ABC):
    """Base class for all PdM models.

    Four concrete families:
        AnomalyDetector  — unsupervised deviation detection
        FailureClassifier — binary/multi-label classification
        RULPredictor     — remaining useful life regression
        SurvivalPredictor — time-to-event with censoring
    """

    formulation: str

    @abstractmethod
    def train(self, train_df: pd.DataFrame, test_df: pd.DataFrame, **kwargs) -> TrainResult:
        ...

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> PredictionResult:
        ...

    @abstractmethod
    def explain(self, features: pd.DataFrame, top_k: int = 10) -> list[dict]:
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "PDMModel":
        ...

    @classmethod
    def detect_formulation(cls, df: pd.DataFrame) -> str:
        """Auto-detect formulation from column patterns."""
        label_cols = [c for c in df.columns if c.startswith("label_")]
        if label_cols:
            return "multilabel"
        if "RUL" in df.columns:
            return "rul"
        if "machine_failure" in df.columns:
            return "classification"
        if "duration" in df.columns and "event" in df.columns:
            return "survival"
        raise ValueError(
            "Cannot detect formulation. Need: label_* columns, 'RUL', "
            "'machine_failure', or 'duration'+'event'"
        )

    @classmethod
    def get_model_class(cls, formulation: str) -> type["PDMModel"]:
        """Route formulation string to the correct model class."""
        from pdm.anomaly_detection.model import AnomalyDetector
        from pdm.fault_prediction.model import FailureClassifier
        from pdm.rul.model import RULPredictor
        from pdm.survival.model import SurvivalPredictor

        registry: dict[str, type[PDMModel]] = {
            "anomaly_detection": AnomalyDetector,
            "classification": FailureClassifier,
            "multilabel": FailureClassifier,
            "rul": RULPredictor,
            "survival": SurvivalPredictor,
        }
        if formulation not in registry:
            raise ValueError(f"Unknown formulation: {formulation}. Valid: {list(registry)}")
        return registry[formulation]
