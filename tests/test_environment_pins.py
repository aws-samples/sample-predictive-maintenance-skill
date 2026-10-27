"""Training and serving environments must pin the same Python and AutoGluon.

``TabularPredictor.load`` defaults to ``require_version_match=True`` and
``require_py_version_match=True``: a model trained with one AutoGluon or Python
minor version cannot be loaded by another. These tests keep every place that
installs AutoGluon tied to the single pin in ``pyproject.toml``.
"""

import re
import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _pinned_autogluon() -> str:
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    pins = [d for d in extras["tabular"] if d.startswith("autogluon.tabular")]
    assert len(pins) == 1, pins
    match = re.fullmatch(r"autogluon\.tabular\[[^\]]+\]==(\d+\.\d+\.\d+)", pins[0])
    assert match, f"AutoGluon must be pinned exactly: {pins[0]}"
    return match.group(1)


def _locked(name: str) -> str:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    versions = {p["version"] for p in lock["package"] if p["name"] == name}
    assert len(versions) == 1, (name, versions)
    return versions.pop()


def test_lock_matches_autogluon_pin():
    pin = _pinned_autogluon()
    assert _locked("autogluon-tabular") == pin
    assert _locked("autogluon-core") == pin


def test_installed_autogluon_matches_pin():
    assert version("autogluon.tabular") == _pinned_autogluon()


def test_container_python_matches_python_version_file():
    pinned = (ROOT / ".python-version").read_text().strip()
    dockerfile = (ROOT / "sagemaker_container" / "Dockerfile").read_text()
    match = re.search(r"^FROM python:(\d+\.\d+)", dockerfile, re.MULTILINE)
    assert match and match.group(1) == pinned


def test_container_installs_locked_serve_extra():
    """The image installs uv.lock's versions; it must not pin anything itself."""
    dockerfile = (ROOT / "sagemaker_container" / "Dockerfile").read_text()
    assert "uv export --frozen --no-dev --no-emit-project --extra serve" in dockerfile
    assert "--require-hashes" in dockerfile
    install_lines = [l for l in dockerfile.splitlines() if "pip install" in l]
    assert install_lines and not any("==" in l for l in install_lines)
    assert not (ROOT / "sagemaker_container" / "requirements.txt").exists()


def test_remote_training_requirements_pin_local_autogluon():
    from pdm.remote.requirements import training_requirements

    reqs = training_requirements().splitlines()
    assert f"autogluon.tabular[lightgbm,catboost,xgboost]=={_pinned_autogluon()}" in reqs
    assert not any(r.startswith("autogluon") and "==" not in r for r in reqs)


@pytest.mark.parametrize("path", ["scripts/setup.sh"])
def test_project_setup_copies_the_lock(path):
    text = (ROOT / path).read_text()
    for name in ("pyproject.toml", "uv.lock", ".python-version"):
        assert name in text


def test_edge_component_installs_pinned_autogluon():
    """Edge loads classification/RUL models via pdm, which imports AutoGluon."""
    reqs = (ROOT / "edge_component" / "requirements.txt").read_text().splitlines()
    assert f"autogluon.tabular[lightgbm,catboost,xgboost]=={_pinned_autogluon()}" in reqs
