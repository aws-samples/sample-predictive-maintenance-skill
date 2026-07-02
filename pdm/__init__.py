"""PDM — Predictive Maintenance modelling library.

Five model families:
    AnomalyDetector          — unsupervised deviation detection (Isolation Forest)
    TemporalAnomalyDetector  — temporal anomaly detection (sliding-window PCA reconstruction)
    FailureClassifier        — binary/multi-label classification (AutoGluon)
    RULPredictor             — remaining useful life regression (sliding window + AutoGluon)
    SurvivalPredictor        — time-to-event with censoring (Cox PH, Weibull AFT, RSF)
"""

from pdm.base import PDMModel, TrainResult, PredictionResult
from pdm.anomaly_detection.model import AnomalyDetector
from pdm.anomaly_detection.temporal import TemporalAnomalyDetector
from pdm.fault_prediction.model import FailureClassifier
from pdm.rul.model import RULPredictor
from pdm.survival.model import SurvivalPredictor
from pdm.data.dataset_schema import DatasetMeta

__all__ = [
    "PDMModel",
    "TrainResult",
    "PredictionResult",
    "AnomalyDetector",
    "TemporalAnomalyDetector",
    "FailureClassifier",
    "RULPredictor",
    "SurvivalPredictor",
    "DatasetMeta",
]
