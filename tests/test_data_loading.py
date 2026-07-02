"""Tests for data loading utilities: DatasetMeta, benchmark_loaders, load_or_cache."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pdm.data.dataset_schema import DatasetMeta


class TestDatasetMeta:
    def test_save_load_roundtrip(self, tmp_path):
        meta = DatasetMeta(
            name="Test", source="benchmark", formulation="rul",
            target_columns=["RUL"], feature_columns=["s1", "s2"],
            n_train=100, n_test=50, n_features=2,
            entity_column="unit_id", time_column="cycle",
        )
        path = tmp_path / "meta.json"
        meta.save(path)
        assert path.exists()

        loaded = DatasetMeta.load(path)
        assert loaded.name == "Test"
        assert loaded.formulation == "rul"
        assert loaded.n_train == 100
        assert loaded.feature_columns == ["s1", "s2"]

    def test_validate_passes_with_correct_csv(self, tmp_path):
        train_df = pd.DataFrame({"RUL": [10, 20], "s1": [1, 2], "s2": [3, 4]})
        train_path = tmp_path / "train.csv"
        train_df.to_csv(train_path, index=False)

        meta = DatasetMeta(
            name="Test", source="benchmark", formulation="rul",
            target_columns=["RUL"], feature_columns=["s1", "s2"],
            n_train=2, n_test=0, n_features=2,
            data_path={"train": str(train_path), "test": str(train_path)},
        )
        meta.validate()  # Should not raise

    def test_validate_raises_on_missing_column(self, tmp_path):
        train_df = pd.DataFrame({"x": [1], "y": [2]})
        train_path = tmp_path / "train.csv"
        train_df.to_csv(train_path, index=False)

        meta = DatasetMeta(
            name="Test", source="benchmark", formulation="rul",
            target_columns=["RUL"], feature_columns=["s1"],
            n_train=1, n_test=0, n_features=1,
            data_path={"train": str(train_path), "test": str(train_path)},
        )
        with pytest.raises(ValueError, match="Target column 'RUL' not in"):
            meta.validate()

    def test_json_is_valid(self, tmp_path):
        meta = DatasetMeta(
            name="X", source="s3", formulation="classification",
            target_columns=["machine_failure"], feature_columns=["f1"],
            n_train=10, n_test=5, n_features=1,
        )
        path = tmp_path / "m.json"
        meta.save(path)
        data = json.loads(path.read_text())
        assert data["name"] == "X"
        assert data["source"] == "s3"


class TestBenchmarkLoaders:
    def test_cmapss_detection(self, tmp_path):
        from pdm.benchmarks.loaders import _is_cmapss
        (tmp_path / "train_FD001.txt").touch()
        assert _is_cmapss(tmp_path) is True

    def test_cmapss_not_detected_on_empty(self, tmp_path):
        from pdm.benchmarks.loaders import _is_cmapss
        assert _is_cmapss(tmp_path) is False

    def test_detect_and_load_returns_none_on_unknown(self, tmp_path):
        from pdm.benchmarks.loaders import detect_and_load
        assert detect_and_load(tmp_path) is None


class TestLoadOrCache:
    def test_creates_parquet_cache(self, tmp_path):
        from pdm.data.utils import load_or_cache
        call_count = [0]

        def loader():
            call_count[0] += 1
            return pd.DataFrame({"x": [1, 2, 3]})

        result = load_or_cache("test_data", loader, cache_dir=str(tmp_path))
        assert len(result) == 3
        assert call_count[0] == 1
        assert (tmp_path / "_cache_test_data.parquet").exists()

    def test_cache_hit_skips_loader(self, tmp_path):
        from pdm.data.utils import load_or_cache
        call_count = [0]

        def loader():
            call_count[0] += 1
            return pd.DataFrame({"x": [1, 2, 3]})

        load_or_cache("test_data", loader, cache_dir=str(tmp_path))
        load_or_cache("test_data", loader, cache_dir=str(tmp_path))
        assert call_count[0] == 1
