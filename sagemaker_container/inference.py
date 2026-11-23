"""Inference handlers for AutoGluon fault-prediction models on SageMaker.

Supports both multi-label (one predictor per label) and single-label models
produced by ``pdm/fault_prediction/train.py``.

The container runs on the slim pdm install (``pdm-project[serve]``, exact
versions from ``uv.lock``), so:

- models whose pickles reference pdm (e.g. custom F2/F0.5 or NASA scorers) load;
- columns are aligned with ``pdm.data.utils.align_to_model``, the same helper
  batch inference and ``runtime.py`` pipelines use;
- a model package may include its project's ``runtime.py``. The endpoint then
  accepts **raw rows** and runs ``engineer_features`` itself, so callers no
  longer re-implement feature engineering.

Model package layout (extracted to ``/opt/ml/model``)::

    metadata.json
    ag_model/
    code/inference.py
    code/runtime.py      # optional: endpoint accepts raw rows
"""
import importlib.util
import json
import os

import pandas as pd
from autogluon.tabular import TabularPredictor

from pdm.data.utils import align_to_model
from pdm.deployment.serving_io import FeatureSchemaError, parse_request_rows, serialize_json_response

RUNTIME_FILE = "runtime.py"


def _load_engineer_features(model_dir):
    """Return ``engineer_features`` from ``code/runtime.py`` if packaged, else None."""
    path = os.path.join(model_dir, "code", RUNTIME_FILE)
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location("pdm_model_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "engineer_features", None)):
        raise ValueError(f"{path} must define engineer_features(raw_df) -> DataFrame")
    return module.engineer_features


def model_fn(model_dir):
    """Load AutoGluon predictor(s), metadata and the optional feature pipeline."""
    metadata_path = os.path.join(model_dir, "metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)

    model_dict = {
        "feature_names": metadata["feature_names"],
        "metadata_path": metadata_path,
        "engineer_features": _load_engineer_features(model_dir),
    }
    ag_model_path = os.path.join(model_dir, "ag_model")

    # Multi-label: one predictor per label
    if "label_names" in metadata:
        predictors = {}
        for label in metadata["label_names"]:
            label_dir = os.path.join(ag_model_path, label.replace(" ", "_"))
            if os.path.isdir(label_dir):
                predictors[label] = TabularPredictor.load(label_dir, verbosity=0)
        return {**model_dict, "predictors": predictors, "multi_label": True}

    # Single-label
    predictor = TabularPredictor.load(ag_model_path, verbosity=0)
    return {**model_dict, "predictor": predictor, "multi_label": False}


def input_fn(request_body, request_content_type):
    """Deserialize non-empty JSON row records or CSV (MIME parameters allowed)."""
    return parse_request_rows(request_body, request_content_type)


def model_input(input_data, model_dict):
    """Raw or engineered rows -> model-ready features in training column order."""
    features = input_data
    if model_dict.get("engineer_features") is not None:
        features = model_dict["engineer_features"](input_data)
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("Feature engineering must produce a non-empty DataFrame")
    if not features.columns.is_unique:
        raise ValueError("Feature engineering produced duplicate feature columns")
    # align_to_model deliberately fills *partially* missing features with zero.
    # However an entirely unrelated schema would silently become all-zero input.
    expected = model_dict["feature_names"]
    if expected and not set(expected).intersection(features.columns):
        raise FeatureSchemaError(
            "No expected model features found in input; check request columns "
            "and whether code/runtime.py was packaged"
        )
    return align_to_model(features, model_dict["metadata_path"])


def predict_fn(input_data, model_dict):
    """Run prediction. Returns dict with predictions + probabilities per label.

    Class predictions use each predictor's validation-calibrated decision
    threshold (``predictor.predict``).
    """
    X = model_input(input_data, model_dict)

    if model_dict["multi_label"]:
        results = {}
        for label, predictor in model_dict["predictors"].items():
            preds = predictor.predict(X)
            results[f"{label}_pred"] = preds.values.tolist()
            try:
                proba = predictor.predict_proba(X, as_multiclass=False)
                results[f"{label}_proba"] = proba.values.tolist()
            except Exception:
                pass
        return results

    # Single-label
    predictor = model_dict["predictor"]
    preds = predictor.predict(X)
    result = {"prediction": preds.values.tolist()}
    try:
        proba = predictor.predict_proba(X)
        result["probabilities"] = proba.values.tolist()
    except Exception:
        pass
    return result


def output_fn(result, response_content_type):
    """Serialize strict JSON while preserving the SageMaker prediction contract."""
    return serialize_json_response(result, response_content_type)
