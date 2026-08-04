"""PDM — Predictive Maintenance modelling library (edge-optimized imports).

This is a minimal __init__.py for edge deployment that uses lazy imports
to avoid requiring all dependencies (e.g., autogluon) when only using
a subset of model types (e.g., anomaly detection with sklearn).

The full library __init__.py eagerly imports all model classes, but on
edge devices we only load what's needed at runtime via PDMModel.get_model_class().
"""

from pdm.base import PDMModel, TrainResult, PredictionResult
from pdm.data.dataset_schema import DatasetMeta

__all__ = [
    "PDMModel",
    "TrainResult",
    "PredictionResult",
    "DatasetMeta",
]
