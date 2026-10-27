"""RUL models trained by the training CLI must load in a fresh process.

``train.py`` uses a custom NASA-score AutoGluon scorer as ``eval_metric``.
AutoGluon pickles the scorer into the saved predictor, so its score function
must live in an importable module. Defined inside ``train.py`` and run as a
script (``python pdm/fault_prediction/train.py``), it pickled as
``__main__._nasa_ag_metric`` and the model could not be loaded anywhere else.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pdm.rul.metrics import NASA_AG_SCORER, nasa_scoring

REPO_ROOT = Path(__file__).resolve().parents[1]
RUL_CSV = Path(__file__).parent / "data" / "rul.csv"


def test_nasa_scorer_matches_nasa_scoring():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 15.0, 30.0, 55.0])
    # AutoGluon maximizes, so the scorer reports the negated NASA score.
    assert NASA_AG_SCORER(y_true, y_pred) == pytest.approx(-nasa_scoring(y_true, y_pred))
    assert NASA_AG_SCORER.name == "nasa_score"
    assert NASA_AG_SCORER._score_func.__module__ == "pdm.rul.metrics"


@pytest.mark.timeout(300)
def test_cli_trained_rul_model_loads_in_fresh_process(tmp_path):
    df = pd.read_csv(RUL_CSV)
    units = sorted(df["unit_id"].unique())
    test_units = units[-max(1, len(units) // 5):]
    train_csv, test_csv = tmp_path / "train.csv", tmp_path / "test.csv"
    df[~df["unit_id"].isin(test_units)].to_csv(train_csv, index=False)
    df[df["unit_id"].isin(test_units)].to_csv(test_csv, index=False)
    out = tmp_path / "model"

    # Run as a script, exactly as the README documents.
    train = subprocess.run(
        [sys.executable, "pdm/fault_prediction/train.py", "--train", str(train_csv),
         "--test", str(test_csv), "--output", str(out), "--time-limit", "20",
         "--presets", "medium_quality", "--skip-importance"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=280,
    )
    assert train.returncode == 0, train.stderr[-2000:]

    load = subprocess.run(
        [sys.executable, "-c",
         "import sys; from autogluon.tabular import TabularPredictor\n"
         f"p = TabularPredictor.load({str(out / 'ag_model')!r}, verbosity=0)\n"
         "assert p.eval_metric.name == 'nasa_score', p.eval_metric.name\n"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert load.returncode == 0, load.stderr[-2000:]
