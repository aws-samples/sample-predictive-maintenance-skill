"""Unified SageMaker serving handler for all PDM model families.

Dispatches to the correct model class based on metadata.json formulation field.
Supports: anomaly_detection, classification, multilabel, rul, survival.
"""

import json
from pathlib import Path

import pandas as pd

from pdm.deployment.serving_io import parse_request_rows, serialize_json_response


def model_fn(model_dir: str):
    """Load any PDMModel from saved artifacts."""
    from pdm.base import PDMModel

    meta_path = Path(model_dir) / "metadata.json"
    metadata = json.loads(meta_path.read_text())
    formulation = metadata["formulation"]

    model_class = PDMModel.get_model_class(formulation)
    return model_class.load(Path(model_dir))


def input_fn(request_body: str | bytes, request_content_type: str):
    """Deserialize validated JSON row records or CSV."""
    return parse_request_rows(request_body, request_content_type)


def predict_fn(input_data: pd.DataFrame, model) -> dict:
    """Run prediction via unified PDMModel interface."""
    result = model.predict(input_data)
    return result.predictions.to_dict(orient="records")


def output_fn(result, response_content_type: str) -> str:
    """Serialize to strict JSON without changing prediction field names."""
    return serialize_json_response(result, response_content_type)
