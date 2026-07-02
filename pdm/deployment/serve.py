"""Unified SageMaker serving handler for all PDM model families.

Dispatches to the correct model class based on metadata.json formulation field.
Supports: anomaly_detection, classification, multilabel, rul, survival.
"""

import json
import os
from pathlib import Path

import pandas as pd


def model_fn(model_dir: str):
    """Load any PDMModel from saved artifacts."""
    from pdm.base import PDMModel

    meta_path = Path(model_dir) / "metadata.json"
    metadata = json.loads(meta_path.read_text())
    formulation = metadata["formulation"]

    model_class = PDMModel.get_model_class(formulation)
    return model_class.load(Path(model_dir))


def input_fn(request_body: str, request_content_type: str) -> pd.DataFrame:
    """Deserialize input (JSON or CSV)."""
    if request_content_type == "application/json":
        return pd.DataFrame(json.loads(request_body))
    if request_content_type == "text/csv":
        from io import StringIO
        return pd.read_csv(StringIO(request_body))
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data: pd.DataFrame, model) -> dict:
    """Run prediction via unified PDMModel interface."""
    result = model.predict(input_data)
    return result.predictions.to_dict(orient="records")


def output_fn(result: dict, response_content_type: str) -> str:
    """Serialize output to JSON."""
    if response_content_type == "application/json":
        return json.dumps(result, default=str)
    raise ValueError(f"Unsupported content type: {response_content_type}")
