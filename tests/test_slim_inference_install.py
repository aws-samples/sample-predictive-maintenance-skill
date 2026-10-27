"""Trained models load and predict with only the slim inference install.

The base ``pdm-project`` install plus one deployment extra must be enough to
load every formulation's model and call ``predict()``. Models are trained here
with the full environment; each check then runs in a fresh interpreter in which
every package outside the target install is made unimportable, simulating:

    edge, anomaly detection   base + edge
    edge, classification/RUL  base + edge + tabular
    edge, survival            base + edge + survival

Set ``PDM_SLIM_VENV_TEST=1`` to additionally build real virtual environments
from ``uv.lock`` for these targets (slow, needs network) instead of relying on
import blocking alone.
"""

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).parent / "data"

# Import names of packages that only the full training environment provides.
TRAIN_ONLY = ["shap", "optuna", "imblearn"]
AWS = ["boto3", "botocore", "pyarrow"]
TABULAR = ["autogluon", "lightgbm", "catboost", "xgboost", "matplotlib"]
SURVIVAL = ["lifelines", "sksurv"]

TARGETS = {
    "edge-anomaly": {"extras": ["edge"], "block": TRAIN_ONLY + AWS + TABULAR + SURVIVAL},
    "edge-tabular": {"extras": ["edge", "tabular"], "block": TRAIN_ONLY + AWS + SURVIVAL},
    "edge-survival": {"extras": ["edge", "survival"], "block": TRAIN_ONLY + AWS + TABULAR},
}

# formulation -> target that must be able to serve it
FORMULATION_TARGET = {
    "anomaly_detection": "edge-anomaly",
    "classification": "edge-tabular",
    "rul_autogluon": "edge-tabular",
    "rul_optuna": "edge-tabular",
    "survival": "edge-survival",
}

_PROBE = textwrap.dedent(
    """
    import json, sys
    block = json.loads(sys.argv[1])
    for name in block:
        sys.modules[name] = None
    import pandas as pd
    from pdm.base import PDMModel

    model_dir, features_csv = sys.argv[2], sys.argv[3]
    meta = json.load(open(f"{model_dir}/metadata.json"))
    model = PDMModel.get_model_class(meta["formulation"]).load(__import__("pathlib").Path(model_dir))
    result = model.predict(pd.read_csv(features_csv))
    print(json.dumps({"rows": len(result.predictions), "columns": list(result.predictions.columns)}))
    """
)


def _train_all(base: Path) -> dict[str, tuple[Path, Path]]:
    """Train one small model per formulation; return {name: (model_dir, features_csv)}."""
    from pdm.anomaly_detection.model import AnomalyDetector
    from pdm.fault_prediction.model import FailureClassifier
    from pdm.rul.model import RULPredictor
    from pdm.survival.model import SurvivalPredictor

    out = {}

    ad = pd.read_csv(DATA / "anomaly.csv").drop(columns=["is_anomaly_label"])
    m = AnomalyDetector(contamination=0.1)
    m.train(ad, ad)
    m.save(base / "anomaly_detection")
    ad.head(5).to_csv(base / "anomaly_detection.csv", index=False)
    out["anomaly_detection"] = (base / "anomaly_detection", base / "anomaly_detection.csv")

    clf = pd.read_csv(DATA / "classification.csv")
    m = FailureClassifier()
    m.train(clf, clf, time_limit=15, presets="medium_quality", output=base / "classification",
            optimization_priority="recall")
    m.save(base / "classification")
    clf.drop(columns=["machine_failure"]).head(5).to_csv(base / "classification.csv", index=False)
    out["classification"] = (base / "classification", base / "classification.csv")

    rul = pd.read_csv(DATA / "rul.csv")
    for backend in ("autogluon", "optuna"):
        name = f"rul_{backend}"
        m = RULPredictor(window_size=5)
        m.train(rul, rul, time_limit=15, presets="medium_quality", backend=backend, n_trials=3,
                output=base / name)
        m.save(base / name)
        rul.to_csv(base / f"{name}.csv", index=False)
        out[name] = (base / name, base / f"{name}.csv")

    surv = pd.read_csv(DATA / "survival.csv")
    m = SurvivalPredictor()
    m.train(surv, surv, time_limit=15, n_trials=2)
    m.save(base / "survival")
    surv.drop(columns=["duration", "event"]).head(5).to_csv(base / "survival.csv", index=False)
    out["survival"] = (base / "survival", base / "survival.csv")
    return out


@pytest.fixture(scope="module")
def trained_models(tmp_path_factory):
    return _train_all(tmp_path_factory.mktemp("slim_models"))


def _probe(python: str, block: list[str], model_dir: Path, features: Path, cwd: Path):
    return subprocess.run(
        [python, "-c", _PROBE, json.dumps(block), str(model_dir), str(features)],
        capture_output=True, text=True, cwd=cwd, timeout=300,
    )


@pytest.mark.timeout(900)
@pytest.mark.parametrize("formulation", sorted(FORMULATION_TARGET))
def test_model_predicts_with_only_its_target_install(formulation, trained_models):
    model_dir, features = trained_models[formulation]
    target = TARGETS[FORMULATION_TARGET[formulation]]
    result = _probe(sys.executable, target["block"], model_dir, features, ROOT)
    assert result.returncode == 0, result.stderr[-3000:]
    assert json.loads(result.stdout.strip().splitlines()[-1])["rows"] > 0


def test_targets_block_nothing_their_extras_install():
    """Guard the simulation: a target never blocks a package it actually installs."""
    installs = {"tabular": set(TABULAR), "survival": set(SURVIVAL), "edge": set()}
    for name, target in TARGETS.items():
        provided = set().union(*(installs[e] for e in target["extras"]))
        assert not provided & set(target["block"]), name


# --- Optional: real virtual environments built from uv.lock ------------------

@pytest.mark.skipif(os.environ.get("PDM_SLIM_VENV_TEST") != "1",
                    reason="set PDM_SLIM_VENV_TEST=1 to build real slim venvs (slow)")
@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("target_name", sorted(TARGETS))
def test_model_predicts_in_real_slim_venv(target_name, trained_models, tmp_path):
    from pdm.deployment.requirements import export_requirements

    target = TARGETS[target_name]
    reqs = tmp_path / "requirements.txt"
    export_requirements(target["extras"], reqs, project_dir=ROOT)
    venv = tmp_path / "venv"
    subprocess.run(["uv", "venv", "--python", sys.executable, str(venv)], check=True,
                   capture_output=True)
    python = str(venv / "bin" / "python")
    subprocess.run(["uv", "pip", "install", "--python", python, "-r", str(reqs)], check=True,
                   capture_output=True)
    subprocess.run(["uv", "pip", "install", "--python", python, "--no-deps", str(ROOT)],
                   check=True, capture_output=True)

    for formulation, tname in FORMULATION_TARGET.items():
        if tname != target_name:
            continue
        model_dir, features = trained_models[formulation]
        # No blocking: the venv only contains what the target installs.
        result = _probe(python, [], model_dir, features, tmp_path)
        assert result.returncode == 0, (formulation, result.stderr[-3000:])
