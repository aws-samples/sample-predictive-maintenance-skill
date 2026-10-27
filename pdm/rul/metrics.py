"""RUL evaluation metrics, including the NASA PHM'08 score as an AutoGluon scorer.

AutoGluon pickles ``eval_metric`` into the saved predictor, so the scorer's
score function must be importable wherever the model is loaded. Keep it in
this module (part of the slim inference install, numpy only) — never define
it in a script, where it would pickle as ``__main__.<name>``.
"""

from __future__ import annotations

import numpy as np


def nasa_scoring(y_true, y_pred) -> float:
    """NASA PHM'08 asymmetric scoring (lower is better).

    Late predictions (over-estimating RUL) are penalized more than early ones.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1).sum())


def _make_nasa_ag_scorer():
    try:
        from autogluon.core.metrics import make_scorer
    except ImportError:  # autogluon is optional for pure-numpy users of nasa_scoring
        return None
    # greater_is_better=False: AutoGluon negates internally and maximizes.
    return make_scorer(
        "nasa_score", nasa_scoring, optimum=0.0, greater_is_better=False, needs_proba=False
    )


NASA_AG_SCORER = _make_nasa_ag_scorer()
