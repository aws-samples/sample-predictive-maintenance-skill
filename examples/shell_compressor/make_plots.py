"""Generate the blog figures from real model outputs + experiment summaries.

Three figures, each one measure per axis (no dual-axis), dataviz palette:
  1. fig_auroc.png     — test AUROC across all five models (identity: AD vs classifier)
  2. fig_roc.png       — ROC curves (real scores): IF, temporal PCA, classifier
  3. fig_tradeoff.png  — small multiples: AUROC vs usable F1 across AD models
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.metrics import roc_curve, roc_auc_score

HERE = Path(__file__).parent
DATA = HERE / "data"
IMG = HERE / "img"
IMG.mkdir(exist_ok=True)

# --- dataviz palette (light mode) ----------------------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 12,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.edgecolor": GRID,
})


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
    ax.tick_params(length=0)


# ===========================================================================
# Figure 1 — AUROC across models (horizontal bars, colored by family)
# ===========================================================================
rows = [  # (label, auroc, family)  family: 0=AD, 1=classifier
    ("Temporal PCA · 1h window", 0.876, 0),
    ("Isolation Forest", 0.866, 0),
    ("Classifier · SMOTE", 0.665, 1),
    ("Classifier · rebalance", 0.643, 1),
    ("Temporal PCA · 2h window", 0.738, 0),
]
rows.sort(key=lambda r: r[1])  # ascending so largest is on top in barh
labels = [r[0] for r in rows]
vals = [r[1] for r in rows]
colors = [BLUE if r[2] == 0 else ORANGE for r in rows]

fig, ax = plt.subplots(figsize=(9.2, 4.6))
y = np.arange(len(rows))
bars = ax.barh(y, vals, height=0.62, color=colors, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=11.5, color=INK)
ax.set_xlim(0, 1.0)
ax.set_ylim(-1.15, len(rows) - 0.1)
ax.set_xlabel("Test AUROC  (ranking quality — higher is better)")
ax.axvline(0.5, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2)
ax.axvline(0.75, color=AQUA, lw=1.4, ls=(0, (4, 3)), zorder=2)
ax.text(0.5, len(rows) - 0.28, "random", color=MUTED, fontsize=9.5, ha="center")
ax.text(0.75, len(rows) - 0.28, "quality gate", color=AQUA, fontsize=9.5, ha="center")
for yi, v in zip(y, vals):
    ax.text(v - 0.012, yi, f"{v:.3f}", va="center", ha="right",
            color="white", fontsize=11, fontweight="bold", zorder=4)
# family legend (identity, not color-alone: labels already name the family)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="Anomaly detection (unsupervised)"),
                   Patch(color=ORANGE, label="Classifier (supervised)")],
          loc="lower center", frameon=False, fontsize=10, ncol=2,
          bbox_to_anchor=(0.5, 0.0), handlelength=1.2, columnspacing=1.8)
ax.set_title("Unsupervised models out-rank the supervised classifier",
             color=INK, fontsize=13, fontweight="bold", pad=22, loc="left")
ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
despine(ax, keep=("left",))
ax.tick_params(axis="y", length=0)
fig.tight_layout()
fig.savefig(IMG / "fig_auroc.png", dpi=160)
plt.close(fig)
print("wrote fig_auroc.png")


# ===========================================================================
# Figure 2 — ROC curves from real scores
# ===========================================================================
test = pd.read_csv(DATA / "raw_test.csv")
y_true = test["machine_failure"].values.astype(int)

curves = []  # (name, color, y_score)

# Isolation Forest baseline — scores already saved in test order
if_scores = pd.read_csv(HERE / "anomaly_detection/baseline/model/anomaly_scores_test.csv")
curves.append(("Isolation Forest  (AUROC 0.87)", BLUE, if_scores["anomaly_score"].values))

# Temporal PCA w6 — load saved model and score
from pdm.anomaly_detection.temporal import TemporalAnomalyDetector
det = TemporalAnomalyDetector.load(HERE / "anomaly_detection/experiments/01_temporal_pca/model")
pca_scores = det.predict(test).predictions["anomaly_score"].values
curves.append(("Temporal PCA · 1h  (AUROC 0.88)", AQUA, pca_scores))

# Classifier SMOTE — predict_proba
from autogluon.tabular import TabularPredictor
pred = TabularPredictor.load(str(HERE / "fault_prediction/experiments/c02_smote/model/ag_model"))
clf_scores = pred.predict_proba(test)[1].values
curves.append(("Classifier · SMOTE  (AUROC 0.67)", ORANGE, clf_scores))

fig, ax = plt.subplots(figsize=(6.6, 6.0))
ax.plot([0, 1], [0, 1], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2)
ax.text(0.62, 0.58, "random", color=MUTED, fontsize=10, rotation=38, ha="center")
for name, color, s in curves:
    fpr, tpr, _ = roc_curve(y_true, s)
    ax.plot(fpr, tpr, color=color, lw=2.4, zorder=3, label=name)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.001)
ax.set_xlabel("False-positive rate")
ax.set_ylabel("True-positive rate")
ax.set_title("Who ranks the compressor trips higher?",
             color=INK, fontsize=13.5, fontweight="bold", pad=12, loc="left")
ax.legend(loc="lower right", frameon=False, fontsize=10.5)
ax.grid(color=GRID, lw=0.8, zorder=0)
despine(ax)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(IMG / "fig_roc.png", dpi=160)
plt.close(fig)
print("wrote fig_roc.png")


# ===========================================================================
# Figure 3 — small multiples: AUROC vs usable F1 (no dual axis)
# ===========================================================================
models = ["Isolation\nForest", "Temporal PCA\n1h window", "Temporal PCA\n2h window"]
auroc = [0.866, 0.876, 0.738]
f1 = [0.00, 0.043, 0.241]
cols = [BLUE, AQUA, AQUA]

fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
for ax, vals, title, hi in (
    (axes[0], auroc, "Ranking quality (AUROC)", 1.0),
    (axes[1], f1, "Usable alarm (point-adjust F1, tuned)", 0.3),
):
    x = np.arange(len(models))
    bars = ax.bar(x, vals, width=0.6, color=cols, zorder=3)
    best = int(np.argmax(vals))
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(b.get_x() + b.get_width() / 2, v + hi * 0.02, f"{v:.3f}" if v else "0.00",
                ha="center", va="bottom", fontsize=10.5,
                fontweight="bold" if i == best else "normal",
                color=INK if i == best else INK2)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, hi)
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold", pad=8, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    despine(ax)
fig.suptitle("A short window ranks best; a long window alarms best",
             color=INK, fontsize=13, fontweight="bold", x=0.01, ha="left", y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(IMG / "fig_tradeoff.png", dpi=160)
plt.close(fig)
print("wrote fig_tradeoff.png")

# sanity: recompute AUROCs from the real scores so the blog numbers are honest
print("recomputed AUROC — IF: %.3f | PCA w6: %.3f | SMOTE: %.3f" % (
    roc_auc_score(y_true, curves[0][2]),
    roc_auc_score(y_true, curves[1][2]),
    roc_auc_score(y_true, curves[2][2]),
))
