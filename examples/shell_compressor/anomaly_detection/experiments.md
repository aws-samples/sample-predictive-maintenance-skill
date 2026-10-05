# Anomaly Detection — Experiments (Shell Compressor)

Baseline: Isolation Forest (point-wise), **AUROC 0.866**, F1 0.0 at p99 threshold.
Quality gate: AUROC ≥ 0.75 (met by baseline). Goal: beat baseline AUROC and
recover a usable operating point (non-zero point-adjust F1).

| # | Hypothesis | Rationale | Status |
|---|-----------|-----------|--------|
| 01 | Temporal PCA reconstruction (sliding window) beats point-wise IF | Data is a 10-min multivariate time series; trips are preceded by contextual drift that windowed reconstruction captures, unlike point-wise IF | **done** |
| 02 | Threshold tuned for point-adjust F1 recovers a usable operating point | Baseline F1=0 is a threshold artifact, not a ranking failure (AUROC 0.87) | **done (within 01)** |

## Results

Reproduce: `python anomaly_detection/run_e01_temporal_pca.py <window> <n_components>`
(n_components as an int forces sklearn's randomized SVD; scaled arrays are kept
float32 — see memory note below).

| Model | Window | AUROC | AUPRC | Point-adjust F1 | Precision | Recall | FP |
|---|---|---|---|---|---|---|---|
| Isolation Forest (baseline) | — | 0.866 | — | 0.00 | — | — | — |
| Temporal PCA | 6 (1h) | **0.876** | 0.013 | 0.043 | 0.022 | 0.667 | 2111 |
| Temporal PCA | 12 (2h) | 0.738 | **0.047** | **0.241** | 0.189 | 0.333 | 103 |

### Findings

- **E01 (temporal context):** A short window (6 steps / 1h) marginally beats the
  point-wise IF baseline on AUROC (0.876 vs 0.866) — temporal reconstruction adds
  little ranking power here because trips are already locally anomalous. A long
  window (12 steps / 2h) *loses* AUROC (0.738) as the windowed subspace smears the
  sharp trip signatures, but produces a far more usable operating point
  (point-adjust F1 0.241, 103 FPs vs 2111).
- **E02 (threshold):** For the AD models the baseline F1=0 *is* largely a
  threshold artifact — a tuned threshold recovers recall of 6/9 segments at w6 and
  a workable F1 at w12. This is the opposite of the classifier, whose weak ranking
  (AUROC ~0.65) cannot be rescued by any threshold (see `fault_prediction/experiments.md`).

### Caveats

- The point-adjust F1 / precision / recall columns use a threshold **optimised on
  the test labels** (`TemporalAnomalyDetector._optimize_threshold`), so they are an
  *optimistic (oracle) ceiling*, not a deployable number. AUROC/AUPRC are
  threshold-independent and leakage-free — judge the models by those.
- No single model dominates: w6 wins ranking, w12 wins the usable operating point.
  For deployment, w12 is the better pick (actionable alarm rate), and AD overall is
  the stronger formulation for this dataset than supervised classification.

### Memory note (8GB box)

`StandardScaler.fit_transform` upcasts to float64, which made the flattened
sliding-window matrix (`n_samples × window·n_features`) blow memory at window=12
(three simultaneous ~3GB copies: windows, reconstruction, difference). Fixed in
`pdm/anomaly_detection/temporal.py` by keeping the scaled arrays float32
(reconstruction error does not need double precision) — this halves the footprint
and also benefits the SMAP/MSL benchmark the detector was written for.

Window = 6/12 steps; scoring=max; smooth=21; n_components=30 (randomized SVD).
