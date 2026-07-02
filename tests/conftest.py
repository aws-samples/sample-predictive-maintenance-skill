"""Test fixtures — load small mock datasets from tests/data/.

All datasets are committed CSV files (< 10 KB each).
No model training in fixtures — tests exercise OUR library code only.
"""
import pandas as pd
import pytest
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def rul_df():
    """5 units × 30 cycles, 4 sensors, RUL target."""
    return pd.read_csv(DATA_DIR / "rul.csv")


@pytest.fixture
def classification_df():
    """100 rows, 5 features, machine_failure (10% positive)."""
    return pd.read_csv(DATA_DIR / "classification.csv")


@pytest.fixture
def survival_df():
    """30 units, duration + event columns."""
    return pd.read_csv(DATA_DIR / "survival.csv")


@pytest.fixture
def anomaly_df():
    """100 rows, 6 sensors, 10% injected spikes. Includes is_anomaly_label for validation."""
    return pd.read_csv(DATA_DIR / "anomaly.csv")


@pytest.fixture
def anomaly_features_df(anomaly_df):
    """Anomaly data without the label column (what the model sees)."""
    return anomaly_df.drop(columns=["is_anomaly_label"])


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Temporary directory for model artifacts."""
    d = tmp_path / "model"
    d.mkdir()
    return d
