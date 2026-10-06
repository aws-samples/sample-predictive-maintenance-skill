"""C03 — decision-threshold tuning for the imbalanced classifier.

Hypothesis: the baseline/rebalanced F1=0.0 is a threshold artifact (all-negative
at the default 0.5 cut), not a ranking failure. We load a trained AutoGluon
predictor, score the test set, and:

  1. Report threshold-independent ranking quality (AUROC, average precision).
  2. Sweep the decision threshold to find the best achievable point-adjust-free
     F1 (an OPTIMISTIC/oracle ceiling, since the threshold sees the test labels).
  3. Pick a threshold on a held-out tail of TRAIN (temporal validation) and apply
     it to TEST for a non-oracle, deployable estimate.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from autogluon.tabular import TabularPredictor

HERE = Path(__file__).parent
DATA = HERE.parent / "data"
MODEL = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "experiments" / "c01_rebalance" / "model" / "ag_model"
OUT = HERE / "experiments" / "c03_threshold"
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "machine_failure"
VAL_FRAC = 0.2  # temporal tail of train used as validation for threshold pick

predictor = TabularPredictor.load(str(MODEL))

train = pd.read_csv(DATA / "raw_train.csv")
test = pd.read_csv(DATA / "raw_test.csv")

y_test = test[TARGET].values.astype(int)
proba_test = predictor.predict_proba(test)[1].values  # P(positive)

auroc = roc_auc_score(y_test, proba_test)
ap = average_precision_score(y_test, proba_test)


def best_threshold(y, p, n=500):
    los, his = np.percentile(p, 50), p.max()
    cands = np.linspace(los, his, n)
    best = (0.0, 0.5, 0.0, 0.0)
    for t in cands:
        yp = (p >= t).astype(int)
        if yp.sum() == 0:
            continue
        f1 = f1_score(y, yp, zero_division=0)
        if f1 > best[0]:
            best = (f1, t, precision_score(y, yp, zero_division=0), recall_score(y, yp, zero_division=0))
    return best


# (1) oracle ceiling: threshold tuned on the test set itself
oracle_f1, oracle_t, oracle_p, oracle_r = best_threshold(y_test, proba_test)

# (2) deployable: tune threshold on temporal tail of TRAIN, apply to TEST
split = int(len(train) * (1 - VAL_FRAC))
val = train.iloc[split:]
y_val = val[TARGET].values.astype(int)
proba_val = predictor.predict_proba(val)[1].values
_, val_t, _, _ = best_threshold(y_val, proba_val)
yp_test = (proba_test >= val_t).astype(int)
dep_f1 = f1_score(y_test, yp_test, zero_division=0)
dep_p = precision_score(y_test, yp_test, zero_division=0)
dep_r = recall_score(y_test, yp_test, zero_division=0)

summary = {
    "experiment": "c03_threshold",
    "model": str(MODEL),
    "auroc": round(float(auroc), 4),
    "average_precision": round(float(ap), 4),
    "default_threshold_0.5": {
        "f1": round(float(f1_score(y_test, (proba_test >= 0.5).astype(int), zero_division=0)), 4),
        "positives_predicted": int((proba_test >= 0.5).sum()),
    },
    "oracle_threshold_on_test": {
        "threshold": round(float(oracle_t), 6), "f1": round(float(oracle_f1), 4),
        "precision": round(float(oracle_p), 4), "recall": round(float(oracle_r), 4),
    },
    "deployable_threshold_from_train_tail": {
        "threshold": round(float(val_t), 6), "f1": round(float(dep_f1), 4),
        "precision": round(float(dep_p), 4), "recall": round(float(dep_r), 4),
    },
    "test_positive_rate": round(float(y_test.mean()), 5),
}
print(json.dumps(summary, indent=2), flush=True)
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"[done] saved to {OUT}", flush=True)
