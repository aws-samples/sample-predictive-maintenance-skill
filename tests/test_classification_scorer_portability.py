"""Models trained with any optimization priority must load without ``pdm``.

AutoGluon pickles the eval_metric into the saved predictor. The SageMaker
serving container (``sagemaker_container/``) installs AutoGluon and sklearn but
not ``pdm``, so a scorer that references a ``pdm`` function makes the model
unloadable at inference time.
"""

import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import fbeta_score

from pdm.fault_prediction.metrics import classification_eval_metric

# Run in a fresh interpreter where importing ``pdm`` raises ImportError,
# simulating the standalone serving container.
_BLOCK_PDM = "import sys; sys.modules['pdm'] = None\n"


def _run_without_pdm(code: str, tmp_path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_PDM + code],
        capture_output=True, text=True, cwd=tmp_path, timeout=120,
    )


@pytest.mark.parametrize("priority", ["balanced", "recall", "precision"])
def test_eval_metric_unpickles_without_pdm(priority, tmp_path):
    metric = classification_eval_metric(priority)
    pkl = tmp_path / "metric.pkl"
    pkl.write_bytes(pickle.dumps(metric))

    result = _run_without_pdm(
        f"import pickle; pickle.loads(open({str(pkl)!r}, 'rb').read())", tmp_path
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(("priority", "beta"), [("recall", 2.0), ("precision", 0.5)])
def test_fbeta_scorers_match_sklearn(priority, beta):
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    scorer = classification_eval_metric(priority)
    expected = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
    assert scorer(y_true, y_pred) == pytest.approx(expected)


def test_recall_priority_predictor_loads_without_pdm(tmp_path):
    from autogluon.tabular import TabularPredictor

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    df["y"] = (df["a"] + rng.normal(scale=0.8, size=300) > 0.8).astype(int)

    model_path = tmp_path / "ag_model"
    TabularPredictor(
        label="y", eval_metric=classification_eval_metric("recall"),
        problem_type="binary", path=str(model_path), verbosity=0,
    ).fit(df, time_limit=30, hyperparameters={"GBM": {}})

    result = _run_without_pdm(
        "from autogluon.tabular import TabularPredictor\n"
        f"p = TabularPredictor.load({str(model_path)!r}, verbosity=0)\n"
        "assert p.eval_metric.name == 'f2'\n",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
