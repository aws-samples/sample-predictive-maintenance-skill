"""Exploratory figure: a mental model of the raw compressor data.

Two stacked blocks in one figure:
  (top)    sensor inventory — how many channels of each kind the compressor exposes
  (bottom) the five physical families' most failure-correlated channel, z-scored,
           across the full timeline, with the nine trips marked.

Everything is computed from benchmark_data/shell_compressor/clean_dataset.csv.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

HERE = Path(__file__).parent
RAW = HERE.parent.parent / "benchmark_data" / "shell_compressor" / "clean_dataset.csv"
IMG = HERE / "img"
IMG.mkdir(exist_ok=True)

ANOMALY_IDX = [10634, 36136, 57280, 57618, 60545, 63144, 118665, 128524, 131118]
HORIZON = 24

# --- palette -------------------------------------------------------------
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
})


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
    ax.tick_params(length=0)


def family(col):
    return re.sub(r"\s*\d+$", "", col).split()[0]


print("loading", RAW)
df = pd.read_csv(RAW)
oi = df["original_index"].values
cols = [c for c in df.columns if c != "original_index"]

# fault-imminent label (same rule as the loader): 24 steps before each trip
label = np.zeros(len(df), dtype=int)
for a in ANOMALY_IDX:
    label[(oi > a - HORIZON) & (oi <= a)] = 1

# --- inventory counts ----------------------------------------------------
PHYS = {"Temperature": "Temperature", "Pressure": "Pressure", "Flow": "Flow rate",
        "Speed": "Speed", "Level": "Level", "Gauging": "Gauging", "Time": "Time ctrl"}
counts, is_phys = {}, {}
for c in cols:
    f = family(c)
    if f in ("Unknown", "Classified"):
        key = "Anonymized"
    else:
        key = PHYS.get(f, f)
    counts[key] = counts.get(key, 0) + 1
    is_phys[key] = key != "Anonymized"
inv = sorted(counts.items(), key=lambda kv: kv[1])  # ascending for barh

# --- most failure-correlated channel per physical family ----------------
want = ["Temperature", "Pressure", "Flow", "Speed", "Level"]
lab = label.astype(float)
lab_c = lab - lab.mean()
picks = []
for fam in want:
    fam_cols = [c for c in cols if family(c) == fam]
    best, best_r = None, -1
    for c in fam_cols:
        x = df[c].values.astype(float)
        x = np.nan_to_num(x, nan=np.nanmean(x))
        xc = x - x.mean()
        denom = np.sqrt((xc**2).sum() * (lab_c**2).sum())
        r = abs((xc * lab_c).sum() / denom) if denom else 0.0
        if r > best_r:
            best, best_r = c, r
    picks.append((fam, best, best_r))

# ========================================================================
fig = plt.figure(figsize=(11, 10.2))
gs = GridSpec(6, 1, height_ratios=[2.1, 1, 1, 1, 1, 1], hspace=0.55,
              left=0.17, right=0.97, top=0.90, bottom=0.07)

# --- panel 1: inventory --------------------------------------------------
ax0 = fig.add_subplot(gs[0])
ypos = np.arange(len(inv))
cvals = [c for _, c in inv]
bar_colors = [ORANGE if k == "Anonymized" else BLUE for k, _ in inv]
ax0.barh(ypos, cvals, color=bar_colors, height=0.68, zorder=3)
ax0.set_yticks(ypos)
ax0.set_yticklabels([k for k, _ in inv], fontsize=10.5, color=INK)
for yi, v in zip(ypos, cvals):
    ax0.text(v + 1.5, yi, str(v), va="center", ha="left", fontsize=10, color=INK2)
ax0.set_xlim(0, max(cvals) * 1.12)
ax0.set_xlabel("number of channels")
ax0.set_title("362 channels: temperature & pressure dominate; a quarter are unlabeled",
              color=INK, fontsize=12.5, fontweight="bold", pad=8, loc="left")
ax0.legend(handles=[Patch(color=BLUE, label="Named physical quantity"),
                    Patch(color=ORANGE, label="Anonymized by Shell")],
           loc="lower right", frameon=False, fontsize=9.5)
ax0.grid(axis="x", color=GRID, lw=0.8, zorder=0)
despine(ax0, keep=("left",))

# --- panels 2-6: representative traces -----------------------------------
# trip x-positions in row space
trip_rows = [int(np.searchsorted(oi, a)) for a in ANOMALY_IDX]
x = np.arange(len(df))
strip_color = {"Temperature": BLUE, "Pressure": ORANGE, "Flow": AQUA,
               "Speed": "#4a3aa7", "Level": "#008300"}
band = max(1, int(len(df) * 0.0025))  # visible half-width for a trip zone
axes = [fig.add_subplot(gs[i + 1]) for i in range(len(picks))]
for k, (ax, (fam, col, r)) in enumerate(zip(axes, picks)):
    v = df[col].values.astype(float)
    v = np.nan_to_num(v, nan=np.nanmean(v))
    z = (v - v.mean()) / (v.std() + 1e-9)
    for tr in trip_rows:  # trip zones behind the trace
        ax.axvspan(tr - band, tr + band, color=RED, alpha=0.16, lw=0, zorder=2)
    ax.plot(x, z, color=strip_color[fam], lw=0.5, zorder=3, rasterized=True)
    ax.set_xlim(0, len(df))
    ax.set_ylim(np.percentile(z, 0.2), np.percentile(z, 99.8))
    ax.set_yticks([])
    ax.set_ylabel(f"{PHYS[fam]}\n(z-score)", rotation=0, ha="right", va="center",
                  fontsize=10.5, color=INK)
    title = f"{col}   ·   |corr with failure| = {r:.2f}"
    if k == 0:
        title += "          red bands = the 9 trips"
    ax.set_title(title, loc="left", fontsize=9, color=INK2, pad=4)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    despine(ax, keep=("bottom",))
    if ax is not axes[-1]:
        ax.set_xticklabels([])
axes[-1].set_xlabel("time  →  (10-minute samples, 106,710 total)")
fig.suptitle("The raw compressor data at a glance", color=INK, fontsize=15,
             fontweight="bold", x=0.17, ha="left", y=0.965)
fig.savefig(IMG / "fig_eda.png", dpi=150, bbox_inches="tight")
print("wrote fig_eda.png")
for fam, col, r in picks:
    print(f"  {fam:12s} -> {col}  (|corr|={r:.3f})")
print(f"  trips at rows: {trip_rows}")
print(f"  positive rows (fault-imminent): {int(label.sum())} / {len(label)} = {label.mean():.4%}")
