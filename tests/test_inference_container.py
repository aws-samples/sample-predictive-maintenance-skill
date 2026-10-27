"""SageMaker container handlers on the slim serving install.

``sagemaker_container/inference.py`` runs in an image with only
``pdm-project[serve]`` installed. These tests train a model with the CLI (using
the recall priority, whose F2 scorer is pickled as a reference to pdm), package
it like Phase 8.3, and invoke the handlers in a fresh interpreter in which
training-only packages are unimportable:

- engineered rows (no runtime.py packaged) — the original contract;
- raw rows with the project's runtime.py packaged — the endpoint runs
  engineer_features itself and must return identical predictions.
"""

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).parent / "data"
INFERENCE_PY = ROOT / "sagemaker_container" / "inference.py"

# Installed by the full training env but not by pdm-project[serve].
NOT_IN_SERVE = ["shap", "optuna", "imblearn", "lifelines", "sksurv"]

RUNTIME_PY = textwrap.dedent(
    """
    import pandas as pd
    from pdm.data.utils import booleans_to_int  # runtime.py may use pdm utilities

    DROP_COLS = ["device_id"]

    def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.drop(columns=[c for c in DROP_COLS if c in raw_df.columns])
        df = booleans_to_int(df, ["door_open"])
        df["sensor_sum"] = df[[c for c in df.columns if c.startswith("sensor_")]].sum(axis=1)
        return df
    """
)

_INVOKE = textwrap.dedent(
    """
    import importlib.util, json, sys
    for name in json.loads(sys.argv[1]):
        sys.modules[name] = None
    spec = importlib.util.spec_from_file_location("container_inference", sys.argv[2])
    inference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference)
    model = inference.model_fn(sys.argv[3])
    data = inference.input_fn(open(sys.argv[4]).read(), "application/json")
    print(inference.output_fn(inference.predict_fn(data, model), "application/json"))
    """
)


def _raw_frame() -> pd.DataFrame:
    df = pd.read_csv(DATA / "classification.csv")
    sensors = [c for c in df.columns if c != "machine_failure"]
    df = df.rename(columns={c: f"sensor_{c}" for c in sensors})
    df.insert(0, "device_id", [f"dev-{i}" for i in range(len(df))])
    df["door_open"] = [i % 3 == 0 for i in range(len(df))]
    return df


@pytest.fixture(scope="module")
def packaged_model(tmp_path_factory):
    base = tmp_path_factory.mktemp("container")
    (base / "runtime.py").write_text(RUNTIME_PY)
    sys.path.insert(0, str(base))
    try:
        from runtime import engineer_features
    finally:
        sys.path.remove(str(base))

    raw = _raw_frame()
    engineered = engineer_features(raw)
    engineered.to_csv(base / "train.csv", index=False)
    engineered.to_csv(base / "test.csv", index=False)

    train = subprocess.run(
        [sys.executable, "pdm/fault_prediction/train.py", "--train", str(base / "train.csv"),
         "--test", str(base / "test.csv"), "--output", str(base / "model"), "--time-limit", "20",
         "--presets", "medium_quality", "--skip-importance", "--optimization-priority", "recall"],
        cwd=ROOT, capture_output=True, text=True, timeout=280,
    )
    assert train.returncode == 0, train.stderr[-2000:]

    # Phase 8.3 packaging
    pkg = base / "model_pkg"
    (pkg / "code").mkdir(parents=True)
    shutil.copytree(base / "model" / "ag_model", pkg / "ag_model")
    shutil.copy(base / "model" / "metadata.json", pkg / "metadata.json")
    shutil.copy(INFERENCE_PY, pkg / "code" / "inference.py")

    rows = raw.drop(columns=["machine_failure"]).head(8)
    (base / "raw.json").write_text(rows.to_json(orient="records"))
    (base / "engineered.json").write_text(
        engineered.drop(columns=["machine_failure"]).head(8).to_json(orient="records")
    )
    return base, pkg


def _invoke(pkg: Path, request: Path) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", _INVOKE, json.dumps(NOT_IN_SERVE), str(pkg / "code" / "inference.py"),
         str(pkg), str(request)],
        capture_output=True, text=True, cwd=pkg, timeout=180,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.timeout(600)
def test_endpoint_serves_engineered_rows_and_raw_rows_identically(packaged_model):
    base, pkg = packaged_model

    engineered = _invoke(pkg, base / "engineered.json")
    assert len(engineered["prediction"]) == 8
    assert len(engineered["probabilities"]) == 8

    shutil.copy(base / "runtime.py", pkg / "code" / "runtime.py")
    try:
        raw = _invoke(pkg, base / "raw.json")
    finally:
        (pkg / "code" / "runtime.py").unlink()

    assert raw == engineered


def test_runtime_without_engineer_features_is_rejected(tmp_path):
    spec = __import__("importlib.util").util.spec_from_file_location("ci", INFERENCE_PY)
    inference = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(inference)
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "runtime.py").write_text("X = 1\n")
    with pytest.raises(ValueError, match="engineer_features"):
        inference._load_engineer_features(str(tmp_path))
