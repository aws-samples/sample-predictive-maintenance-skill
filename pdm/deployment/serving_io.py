"""Input and output contracts shared by PDM and SageMaker serving handlers.

Accept non-empty JSON row arrays or CSV, including standard MIME parameters.
Preserve the established JSON response layouts while producing strict JSON.
"""

import json
import math
from io import StringIO

import numpy as np
import pandas as pd


class FeatureSchemaError(ValueError):
    """The request has no features recognizable by the packaged model."""


def _media_type(content_type: str) -> str:
    if not isinstance(content_type, str) or not content_type.strip():
        raise ValueError("Missing Content-Type")
    return content_type.split(";", 1)[0].strip().lower()


def parse_request_rows(request_body: str | bytes, content_type: str) -> pd.DataFrame:
    """Parse JSON records or CSV as a non-empty table, with clear errors."""
    media_type = _media_type(content_type)
    if media_type not in ("application/json", "text/csv"):
        raise ValueError(f"Unsupported content type: {content_type}")

    if isinstance(request_body, bytes):
        try:
            request_body = request_body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Request body must be UTF-8") from exc
    if not isinstance(request_body, str):
        raise ValueError("Request body must be a string or UTF-8 bytes")

    if media_type == "application/json":
        try:
            rows = json.loads(
                request_body,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"Invalid JSON constant: {token}")
                ),
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON request body: {exc.msg}") from exc
        if isinstance(rows, list):
            if not rows or not all(isinstance(row, dict) and row for row in rows):
                raise ValueError("JSON input must be a non-empty array of row objects")
            frame = pd.DataFrame(rows)
        elif isinstance(rows, dict) and rows and all(
            isinstance(values, list) for values in rows.values()
        ):
            # Preserve the pre-existing DataFrame(JSON) column-array format.
            if len({len(values) for values in rows.values()}) != 1:
                raise ValueError("JSON column arrays must have equal lengths")
            frame = pd.DataFrame(rows)
        else:
            raise ValueError(
                "JSON input must be a non-empty array of row objects "
                "or an object of equal-length column arrays"
            )
    else:
        try:
            frame = pd.read_csv(StringIO(request_body))
        except pd.errors.EmptyDataError as exc:
            raise ValueError("CSV request body must contain a header and data rows") from exc

    if frame.empty or not frame.columns.is_unique:
        raise ValueError("Request must contain data rows with unique feature columns")
    return frame


def _json_safe(value):
    """Preserve output shapes; encode non-finite / missing numbers as JSON null."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def serialize_json_response(result, content_type: str) -> str:
    """Generate standards-compliant JSON, preserving the existing API structure."""
    if _media_type(content_type) != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    return json.dumps(_json_safe(result), allow_nan=False, default=str)
