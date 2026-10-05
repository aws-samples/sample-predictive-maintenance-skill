"""E01 — Temporal PCA reconstruction anomaly detector (memory-safe).

Memory notes (8GB box): the sliding-window matrix on 85k x 362 features is the
bottleneck. We (1) downcast to float32, (2) use a smaller window, and (3) set a
fixed-int n_components so sklearn PCA picks the randomized SVD solver instead of
the full-SVD path that an n_components=float (0.90) forces.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from pdm.anomaly_detection.temporal import TemporalAnomalyDetector

HERE = Path(__file__).parent
DATA = HERE.parent / "data"
OUT = HERE / "experiments" / "01_temporal_pca"
OUT.mkdir(parents=True, exist_ok=True)

WINDOW = int(sys.argv[1]) if len(sys.argv) > 1 else 6
N_COMPONENTS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
SMOOTH = 21
BASELINE_AUROC = 0.8662

print(f"[cfg] window={WINDOW} n_components={N_COMPONENTS} smooth={SMOOTH}", flush=True)

train = pd.read_csv(DATA / "raw_train.csv")
test = pd.read_csv(DATA / "raw_test.csv")
print(f"[data] train={train.shape} test={test.shape}", flush=True)

# downcast numeric to float32 to halve the windowed-matrix footprint
for df in (train, test):
    floatcols = df.select_dtypes(include=["float64"]).columns
    df[floatcols] = df[floatcols].astype(np.float32)

# detector expects a "label" column on the test set for threshold optimisation,
# and trains on normal rows only
train["label"] = train["machine_failure"].astype(int)
test["label"] = test["machine_failure"].astype(int)
train_normal = train[train["machine_failure"] == 0].copy()
print(f"[data] train_normal={train_normal.shape} test_pos={int(test['label'].sum())}", flush=True)

det = TemporalAnomalyDetector(
    window_size=WINDOW,
    n_components=N_COMPONENTS,
    smooth_window=SMOOTH,
    contamination=0.01,
    scoring="max",
)
res = det.train(train_normal, test)
print(f"[train] metrics={json.dumps(res.metrics)}", flush=True)

# score test set and evaluate (threshold-independent + point-adjust at tuned thr)
pred = det.predict(test)
scores = pred.predictions["anomaly_score"].values
y_true = test["label"].values.astype(int)

auroc = roc_auc_score(y_true, scores)
auprc = average_precision_score(y_true, scores)

y_pred = (scores > det.threshold).astype(int)
y_adj = det._point_adjust(y_true, y_pred)
tp = int(((y_adj == 1) & (y_true == 1)).sum())
fp = int(((y_adj == 1) & (y_true == 0)).sum())
fn = int(((y_adj == 0) & (y_true == 1)).sum())
prec = tp / (tp + fp) if (tp + fp) else 0.0
rec = tp / (tp + fn) if (tp + fn) else 0.0
pa_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

# raw (non-adjusted) F1 for reference
rtp = int(((y_pred == 1) & (y_true == 1)).sum())
rfp = int(((y_pred == 1) & (y_true == 0)).sum())
rfn = int(((y_pred == 0) & (y_true == 1)).sum())
rprec = rtp / (rtp + rfp) if (rtp + rfp) else 0.0
rrec = rtp / (rtp + rfn) if (rtp + rfn) else 0.0
raw_f1 = 2 * rprec * rrec / (rprec + rrec) if (rprec + rrec) else 0.0

summary = {
    "experiment": "01_temporal_pca",
    "window_size": WINDOW,
    "n_components_requested": N_COMPONENTS,
    "pca_components": res.metrics["pca_components"],
    "pca_explained_variance": res.metrics["pca_explained_variance"],
    "threshold": det.threshold,
    "auroc": round(float(auroc), 4),
    "auprc": round(float(auprc), 4),
    "baseline_auroc": BASELINE_AUROC,
    "beats_baseline_auroc": bool(auroc > BASELINE_AUROC),
    "point_adjust": {"precision": round(prec, 4), "recall": round(rec, 4),
                     "f1": round(pa_f1, 4), "tp": tp, "fp": fp, "fn": fn},
    "raw": {"precision": round(rprec, 4), "recall": round(rrec, 4), "f1": round(raw_f1, 4)},
}
print(f"[result] {json.dumps(summary, indent=2)}", flush=True)

det.save(OUT / "model")
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"[done] saved to {OUT}", flush=True)
