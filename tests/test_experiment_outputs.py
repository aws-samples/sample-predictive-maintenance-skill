"""Experiment output paths must not silently replace previous model artifacts."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from pdm.experiment_outputs import reserve_experiment_run_dir


def test_creates_run_under_new_experiment_root(tmp_path):
    root = tmp_path / "experiments" / "01_temporal_pca"
    out = reserve_experiment_run_dir(root, "w6_components30")
    assert out.parent == root
    assert out.is_dir()
    assert out.name.startswith("w6_components30_")


def test_repeated_configuration_never_overwrites_models(tmp_path):
    root = tmp_path / "01_temporal_pca"
    first = reserve_experiment_run_dir(root, "w6_components30")
    (first / "model").mkdir()
    (first / "model" / "metadata.json").write_text('{"window": 6}')
    (first / "summary.json").write_text('{"auroc": 0.87}')

    second = reserve_experiment_run_dir(root, "w6_components30")
    (second / "summary.json").write_text('{"auroc": 0.70}')
    assert first != second
    assert (first / "model" / "metadata.json").read_text() == '{"window": 6}'
    assert (first / "summary.json").read_text() == '{"auroc": 0.87}'
    assert (second / "summary.json").read_text() == '{"auroc": 0.70}'


def test_different_window_configs_have_traceable_paths(tmp_path):
    root = tmp_path / "01_temporal_pca"
    a = reserve_experiment_run_dir(root, "w6_components30")
    b = reserve_experiment_run_dir(root, "w12_components30")
    assert a != b
    assert "w6_components30" in a.name
    assert "w12_components30" in b.name


def test_run_reservation_is_atomic_under_concurrency(tmp_path):
    root = tmp_path / "runs"
    with ThreadPoolExecutor(max_workers=8) as executor:
        dirs = list(executor.map(
            lambda _: reserve_experiment_run_dir(root, "same_config"), range(16)
        ))
    assert len(set(dirs)) == 16
    assert all(path.is_dir() for path in dirs)


@pytest.mark.parametrize("variant", ["", "../outside", "w6/x", ".", "space name", "bad\\\\name", "ä", "x" * 81])
def test_rejects_unsafe_run_variant(tmp_path, variant):
    with pytest.raises(ValueError, match="variant must"):
        reserve_experiment_run_dir(tmp_path / "runs", variant)
    assert not (tmp_path / "runs").exists()


def test_preserves_legacy_model_directory(tmp_path):
    root = tmp_path / "01_temporal_pca"
    legacy = root / "model"
    legacy.mkdir(parents=True)
    (legacy / "pca_model.joblib").write_bytes(b"old model")
    new_dir = reserve_experiment_run_dir(root, "w12_components30")
    assert new_dir != root
    assert (legacy / "pca_model.joblib").read_bytes() == b"old model"
