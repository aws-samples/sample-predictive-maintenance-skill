"""PDM — Predictive Maintenance modelling library.

Five model families:
    AnomalyDetector          — unsupervised deviation detection (Isolation Forest)
    TemporalAnomalyDetector  — temporal anomaly detection (sliding-window PCA reconstruction)
    FailureClassifier        — binary/multi-label classification (AutoGluon)
    RULPredictor             — remaining useful life regression (sliding window + AutoGluon)
    SurvivalPredictor        — time-to-event with censoring (Cox PH, Weibull AFT, RSF)

Imports are lazy: model classes are loaded on first access to avoid requiring
all dependencies (autogluon, lifelines, etc.) when only one formulation is used.
"""

from pdm import solution_user_agent  # noqa: F401 - registers the AWS Solutions user-agent hook; import first
from pdm.base import PDMModel, TrainResult, PredictionResult
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

# Lazy imports — model classes are only loaded when accessed by name.
# This avoids importing autogluon/lifelines/etc. at package import time.
_LAZY_IMPORTS = {
    "AnomalyDetector": ("pdm.anomaly_detection.model", "AnomalyDetector"),
    "TemporalAnomalyDetector": ("pdm.anomaly_detection.temporal", "TemporalAnomalyDetector"),
    "FailureClassifier": ("pdm.fault_prediction.model", "FailureClassifier"),
    "RULPredictor": ("pdm.rul.model", "RULPredictor"),
    "SurvivalPredictor": ("pdm.survival.model", "SurvivalPredictor"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, class_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        # Cache it on the module so __getattr__ isn't called again
        globals()[name] = cls
        return cls
    raise AttributeError(f"module 'pdm' has no attribute {name!r}")
