"""Inference handlers for AutoGluon TabularPredictor on SageMaker.

Supports both multi-label (one predictor per label) and single-label models.
Must be standalone — no imports from local training packages.
"""
import json
import os

import pandas as pd
from autogluon.tabular import TabularPredictor


def model_fn(model_dir):
    """Load AutoGluon predictor(s) + metadata from model_dir."""
    with open(os.path.join(model_dir, "metadata.json")) as f:
        metadata = json.load(f)

    feature_names = metadata["feature_names"]
    ag_model_path = os.path.join(model_dir, "ag_model")

    # Multi-label: one predictor per label
    if "label_names" in metadata:
        label_names = metadata["label_names"]
        predictors = {}
        for label in label_names:
            label_dir = os.path.join(ag_model_path, label.replace(" ", "_"))
            if os.path.isdir(label_dir):
                predictors[label] = TabularPredictor.load(label_dir, verbosity=0)
        return {"predictors": predictors, "feature_names": feature_names, "multi_label": True}

    # Single-label
    predictor = TabularPredictor.load(ag_model_path, verbosity=0)
    return {"predictor": predictor, "feature_names": feature_names, "multi_label": False}


def input_fn(request_body, request_content_type):
    """Deserialize input (JSON array of objects or CSV)."""
    if request_content_type == "application/json":
        return pd.DataFrame(json.loads(request_body))
    if request_content_type == "text/csv":
        from io import StringIO
        return pd.read_csv(StringIO(request_body))
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model_dict):
    """Run prediction. Returns dict with predictions + probabilities per label."""
    feature_names = model_dict["feature_names"]
    X = input_data.reindex(columns=feature_names, fill_value=0)

    if model_dict["multi_label"]:
        results = {}
        for label, predictor in model_dict["predictors"].items():
            preds = predictor.predict(X)
            results[f"{label}_pred"] = preds.values.tolist()
            try:
                proba = predictor.predict_proba(X).iloc[:, 1]
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
    """Serialize output to JSON."""
    if response_content_type == "application/json":
        return json.dumps(result)
    raise ValueError(f"Unsupported content type: {response_content_type}")
