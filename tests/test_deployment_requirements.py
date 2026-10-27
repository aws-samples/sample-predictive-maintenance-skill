"""Dependency split: slim base install, per-target extras, locked exports."""

import shutil
import tomllib
from pathlib import Path

import pytest

from pdm.deployment.requirements import (
    FORMULATION_EXTRAS,
    export_requirements,
    extras_for,
)

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())

TRAINING_ONLY = ("autogluon", "lifelines", "scikit-survival", "shap", "optuna",
                 "imbalanced-learn", "matplotlib", "boto3", "pyarrow")


def test_base_install_has_no_training_or_aws_dependencies():
    for dep in PYPROJECT["project"]["dependencies"]:
        assert not dep.lower().startswith(TRAINING_ONLY), dep


def test_default_dev_group_installs_full_training_environment():
    assert PYPROJECT["dependency-groups"]["dev"] == ["pdm-project[test]"]
    extras = PYPROJECT["project"]["optional-dependencies"]
    assert "pdm-project[train]" in extras["test"]
    assert "pdm-project[tabular,survival,aws]" in extras["train"]


@pytest.mark.parametrize(
    ("target", "formulation", "expected"),
    [
        ("edge", "anomaly_detection", ["edge"]),
        ("edge", "classification", ["edge", "tabular"]),
        ("edge", "multi-label classification", ["edge", "tabular"]),
        ("edge", "rul", ["edge", "tabular"]),
        ("edge", "regression", ["edge", "tabular"]),
        ("edge", "survival", ["edge", "survival"]),
        ("serve", None, ["serve"]),
        ("serve", "classification", ["serve", "tabular"]),
    ],
)
def test_extras_for_target_and_formulation(target, formulation, expected):
    assert extras_for(target, formulation) == expected


def test_every_formulation_extra_exists():
    defined = set(PYPROJECT["project"]["optional-dependencies"])
    for extras in FORMULATION_EXTRAS.values():
        assert set(extras) <= defined


def test_unknown_formulation_is_rejected():
    with pytest.raises(ValueError, match="Unknown formulation"):
        extras_for("edge", "forecasting")


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")
@pytest.mark.parametrize(
    ("extras", "present", "absent"),
    [
        (["edge"], ["scikit-learn==", "awsiotsdk=="], ["autogluon", "lifelines", "boto3", "shap"]),
        (["edge", "tabular"], ["autogluon-tabular==", "catboost=="], ["lifelines", "shap", "optuna"]),
        (["serve"], ["autogluon-tabular==", "flask==", "boto3=="], ["lifelines", "shap", "optuna", "imbalanced-learn"]),
    ],
)
def test_export_is_locked_and_slim(extras, present, absent, tmp_path):
    text = export_requirements(extras, tmp_path / "req.txt", project_dir=ROOT).read_text()
    lines = [line.strip() for line in text.splitlines()]
    names = [line for line in lines if line and not line.startswith(("#", "--hash"))]
    for prefix in present:
        assert any(n.startswith(prefix) for n in names), prefix
    for prefix in absent:
        assert not any(n.startswith(prefix) for n in names), prefix
    assert "--hash=sha256:" in text  # pip installs in hash-checking mode
    assert not any(n.startswith("pdm-project") or n.startswith("-e") for n in names)
