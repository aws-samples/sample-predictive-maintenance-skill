"""RULPredictor must serve the predictions it was evaluated with, after save/load.

Regressions covered:
- predict() on raw time series skipped the per-sensor normalization applied in
  train(), so served predictions disagreed with the evaluated model.
- the optuna backend's estimator was never saved, so load() failed.
"""

import numpy as np
import pandas as pd
import pytest

from pdm.rul.model import RULPredictor


def _last_true_rul(df: pd.DataFrame, cap: int) -> np.ndarray:
    return df.groupby("unit_id").tail(1)["RUL"].clip(upper=cap).values


@pytest.mark.timeout(300)
def test_predict_on_raw_series_matches_training_evaluation(rul_df, tmp_path):
    model = RULPredictor(window_size=5)
    result = model.train(rul_df, rul_df, time_limit=15, presets="medium_quality",
                         output=tmp_path / "train_out")
    served = model.predict(rul_df).predictions["predicted_rul"].values
    rmse = float(np.sqrt(np.mean((_last_true_rul(rul_df, model.rul_cap) - served) ** 2)))
    # train() evaluates on the last window of each test unit; predict() must agree.
    assert rmse == pytest.approx(result.metrics["rmse"], abs=0.01)


@pytest.mark.timeout(300)
@pytest.mark.parametrize("backend", ["autogluon", "optuna"])
def test_save_load_roundtrip_reproduces_predictions(backend, rul_df, tmp_path):
    model = RULPredictor(window_size=5)
    model.train(rul_df, rul_df, time_limit=15, presets="medium_quality", backend=backend,
                n_trials=3, output=tmp_path / "train_out")
    expected = model.predict(rul_df).predictions["predicted_rul"].values

    # Save somewhere other than the training output dir.
    model.save(tmp_path / "saved")
    loaded = RULPredictor.load(tmp_path / "saved")

    assert loaded.backend == backend
    np.testing.assert_allclose(
        loaded.predict(rul_df).predictions["predicted_rul"].values, expected, rtol=1e-6
    )
