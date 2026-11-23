"""SageMaker input/output contract tests for both serving entry points."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pdm.deployment import serve as unified
from pdm.deployment.serving_io import FeatureSchemaError
from pdm.base import PDMModel

ROOT = Path(__file__).resolve().parents[1]
# The SageMaker container is copied into /opt/program, not installed as a
# Python package. Load the same entry point by its on-disk path.
spec = importlib.util.spec_from_file_location(
    "sagemaker_contract_inference", ROOT / "sagemaker_container" / "inference.py"
)
container = importlib.util.module_from_spec(spec)
spec.loader.exec_module(container)

HANDLERS = [unified, container]


@pytest.mark.parametrize("handler", HANDLERS, ids=["unified", "container"])
@pytest.mark.parametrize("body,content_type", [
    ('[{"sensor":1.5},{"sensor":2.5}]', "application/json"),
    ('[{"sensor":1.5}]', "Application/JSON; charset=utf-8"),
    ('{"sensor":[1.5,2.5]}', "application/json"),
    ("sensor\n1.5\n2.5", "text/csv; charset=UTF-8"),
])
def test_valid_json_csv_and_mime_parameters(handler, body, content_type):
    frame = handler.input_fn(body, content_type)
    assert list(frame.columns) == ["sensor"]
    assert len(frame) in (1, 2)


@pytest.mark.parametrize("handler", HANDLERS, ids=["unified", "container"])
def test_accept_utf8_byte_body(handler):
    frame = handler.input_fn(b'[{"sensor":3}]', "application/json; charset=utf-8")
    assert frame.iloc[0]["sensor"] == 3


@pytest.mark.parametrize("handler", HANDLERS, ids=["unified", "container"])
@pytest.mark.parametrize("body,content_type,error", [
    ("[", "application/json", "Invalid JSON request"),
    ("[]", "application/json", "non-empty array"),
    ('{"sensor":1}', "application/json", "array of row objects"),
    ("[1,2]", "application/json", "array of row objects"),
    ("[{}]", "application/json", "array of row objects"),
    ('{"a":[1],"b":[1,2]}', "application/json", "equal lengths"),
    ('[{"sensor":NaN}]', "application/json", "Invalid JSON constant"),
    ("", "text/csv", "CSV request body"),
    ("sensor\n", "text/csv", "data rows"),
    ("sensor\n1", "application/octet-stream", "Unsupported content type"),
])
def test_bad_input_fails_explicitly(handler, body, content_type, error):
    with pytest.raises(ValueError, match=error):
        handler.input_fn(body, content_type)


@pytest.mark.parametrize("handler", HANDLERS, ids=["unified", "container"])
def test_missing_content_type_and_non_utf8_body(handler):
    with pytest.raises(ValueError, match="Missing Content-Type"):
        handler.input_fn('[]', None)
    with pytest.raises(ValueError, match="UTF-8"):
        handler.input_fn(b"\xff", "application/json")


@pytest.mark.parametrize("handler", HANDLERS, ids=["unified", "container"])
def test_output_strict_json_with_missing_numbers(handler):
    record = {"prediction": [np.int64(1), np.float32(np.nan), np.inf, -np.inf, pd.NA],
              "probabilities": [[np.float32(0.25), np.float64(0.75)]]}
    text = handler.output_fn(record, "application/json; charset=UTF-8")
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {
        "prediction": [1, None, None, None, None],
        "probabilities": [[0.25, 0.75]],
    }


@pytest.mark.parametrize("handler", HANDLERS, ids=["unified", "container"])
def test_response_rejects_unknown_media_type(handler):
    with pytest.raises(ValueError, match="Unsupported content type"):
        handler.output_fn({"prediction": []}, "text/csv")


def test_unified_model_dispatch_for_all_formulations(tmp_path, monkeypatch):
    metadata_path = tmp_path / "metadata.json"
    invoked = []

    class MockModel:
        @classmethod
        def load(cls, path):
            invoked.append(path)
            return "loaded model"

    monkeypatch.setattr(
        PDMModel, "get_model_class",
        classmethod(lambda cls, formulation: MockModel),
    )
    for formulation in ["anomaly_detection", "classification", "multilabel", "rul", "survival"]:
        metadata_path.write_text(json.dumps({"formulation": formulation}))
        assert unified.model_fn(str(tmp_path)) == "loaded model"
    assert invoked == [tmp_path] * 5


def test_container_predict_keeps_single_and_multilabel_response_keys(tmp_path, monkeypatch):
    class MockPredictor:
        def predict(self, X):
            return pd.Series([1] * len(X))
        def predict_proba(self, X, as_multiclass=True):
            if as_multiclass:
                return pd.DataFrame([[0.2, 0.8]] * len(X))
            return pd.Series([0.8] * len(X))

    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"feature_names": ["sensor"]}))
    model_dict = {"feature_names": ["sensor"], "metadata_path": str(metadata),
                  "engineer_features": None, "predictor": MockPredictor(), "multi_label": False}
    rows = pd.DataFrame({"sensor": [1.0]})
    assert container.predict_fn(rows, model_dict) == {
        "prediction": [1], "probabilities": [[0.2, 0.8]],
    }
    model_dict = {**model_dict, "multi_label": True, "predictors": {"failure": MockPredictor()}}
    assert container.predict_fn(rows, model_dict) == {
        "failure_pred": [1], "failure_proba": [0.8],
    }


def test_container_rejects_unrelated_schema_but_zero_fills_partial_features(tmp_path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"feature_names": ["a", "b"]}))
    model_dict = {"feature_names": ["a", "b"], "metadata_path": str(metadata),
                  "engineer_features": None}
    aligned = container.model_input(pd.DataFrame({"a": [3]}), model_dict)
    assert list(aligned.columns) == ["a", "b"]
    assert aligned["b"].tolist() == [0]
    with pytest.raises(FeatureSchemaError, match="No expected model features"):
        container.model_input(pd.DataFrame({"unrelated": [1]}), model_dict)


def test_container_rejects_invalid_runtime_output(tmp_path):
    model_dict = {"feature_names": ["a"], "metadata_path": str(tmp_path / "unused.json"),
                  "engineer_features": lambda df: None}
    with pytest.raises(ValueError, match="non-empty DataFrame"):
        container.model_input(pd.DataFrame({"a": [1]}), model_dict)


def test_http_invalid_request_returns_400_without_loading(monkeypatch):
    flask = pytest.importorskip("flask")
    monkeypatch.setitem(sys.modules, "inference", container)
    source = ROOT / "sagemaker_container" / "serve.py"
    spec = importlib.util.spec_from_file_location("pdm_http_contract_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def model_should_not_load(path):
        raise AssertionError("invalid input must not cold-load the model")
    monkeypatch.setattr(module, "model_fn", model_should_not_load)
    response = module.app.test_client().post(
        "/invocations", data='{"sensor":1}', content_type="application/json; charset=utf-8"
    )
    assert response.status_code == 400
    assert "array of row objects" in response.get_json()["error"]
    assert module.model is None


def test_http_valid_parameterized_content_type_returns_200(monkeypatch):
    pytest.importorskip("flask")
    monkeypatch.setitem(sys.modules, "inference", container)
    source = ROOT / "sagemaker_container" / "serve.py"
    spec = importlib.util.spec_from_file_location("pdm_http_contract_valid", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "model_fn", lambda path: object())
    monkeypatch.setattr(module, "predict_fn", lambda df, model: {"prediction": [1] * len(df)})
    response = module.app.test_client().post(
        "/invocations", data='[{"sensor":1}]', content_type="application/json; charset=utf-8"
    )
    assert response.status_code == 200
    assert response.get_json() == {"prediction": [1]}


def test_http_unrelated_schema_returns_400_not_500(monkeypatch):
    pytest.importorskip("flask")
    monkeypatch.setitem(sys.modules, "inference", container)
    source = ROOT / "sagemaker_container" / "serve.py"
    spec = importlib.util.spec_from_file_location("pdm_http_contract_schema", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "model_fn", lambda path: object())

    def bad_schema(df, model):
        raise FeatureSchemaError("No expected model features found in input")

    monkeypatch.setattr(module, "predict_fn", bad_schema)
    response = module.app.test_client().post(
        "/invocations", data='[{"wrong":1}]', content_type="application/json"
    )
    assert response.status_code == 400
    assert "No expected model features" in response.get_json()["error"]
