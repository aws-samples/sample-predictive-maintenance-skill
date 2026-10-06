# Classification — Experiments (Shell Compressor)

Baseline: AutoGluon `medium_quality`, temporal 80/20 split, no rebalancing.
**Test F1 = 0.00** (collapses to all-negative; validation F1 ≈ 0.67).
Quality gate: F1 ≥ 0.50. Positive rate: ~0.17% train / ~0.34% test.

| # | Hypothesis | Lever | Status |
|---|-----------|-------|--------|
| C01 | Inverse-frequency sample weights make the minority class learnable | `--rebalance` | **done** |
| C02 | SMOTE oversampling creates a learnable minority manifold | `--smote` | **done** |
| C03 | Baseline F1=0 is a threshold artifact, not a ranking failure | decision-threshold tuning | **done** |

## Results

| Experiment | Default-0.5 F1 | Test AUROC | Avg. precision | Oracle-threshold F1 | Deployable-threshold F1 |
|---|---|---|---|---|---|
| Baseline (none) | 0.00 | — | — | — | — |
| C01 `--rebalance` | 0.00 | 0.643 | 0.007 | 0.028 | 0.00 |
| C02 `--smote` | 0.00 | 0.665 | 0.013 | 0.060 | 0.00 |

- **Oracle-threshold F1** = best F1 over all thresholds on the *test* set — an
  optimistic ceiling (sees test labels).
- **Deployable-threshold F1** = threshold picked on a temporal tail of *train*,
  applied to test. Both experiments pick 0.5 (the validation tail has no usable
  positive signal), so deployable F1 = 0.

### Findings

- **Neither rebalancing nor SMOTE clears the gate.** Both still predict
  all-negative at the default threshold (F1 0.00). SMOTE edges ahead of inverse-
  frequency weighting on ranking (AUROC 0.665 vs 0.643, oracle-F1 0.060 vs 0.028)
  but both remain far below F1 ≥ 0.50.
- **The classifier's problem is ranking, not threshold.** Even the oracle
  threshold tops out at F1 0.060. Test AUROC ≈ 0.64–0.67 is only marginally better
  than random (0.5). With just 144 positive training rows across a handful of
  distinct trips, the supervised model overfits those specific pre-trip windows
  and does not generalise to the unseen test trips.
- **Cross-formulation conclusion:** the *unsupervised* anomaly detectors (IF 0.866,
  temporal PCA 0.876 AUROC) substantially out-rank the *supervised* classifier
  (0.64–0.67 AUROC) on this dataset. When positives are this sparse and
  trip-specific, "flag the unusual" beats "learn the fault signature." Anomaly
  detection is the recommended formulation here.

## Bug / robustness fixes made while running these

1. **`--rebalance` crashed at `evaluate()`** (pre-existing). The `sample_weight`
   column was added to the training frame but never registered with AutoGluon, so
   it was treated as a predictive feature and `evaluate(test_df)` raised
   `KeyError: ['sample_weight']`. Fixed by passing `sample_weight="sample_weight"`
   to `TabularPredictor(...)` and excluding it from `feature_cols`
   (`pdm/fault_prediction/train.py`).
2. **`--smote` crashed on NaNs** (`ValueError: Input X contains NaN`). SMOTE
   interpolates between neighbours and cannot handle missing values, which this
   SCADA data has. Fixed by median-imputing numeric columns before `fit_resample`
   (all-NaN columns fall back to 0).

Reproduce:
```bash
python ../../pdm/fault_prediction/train.py --train data/raw_train.csv \
  --test data/raw_test.csv --output fault_prediction/experiments/c01_rebalance/model \
  --presets medium_quality --time-limit 300 --rebalance --skip-importance
# swap --rebalance for --smote → c02_smote
python fault_prediction/run_c03_threshold.py <model>/ag_model   # threshold sweep
```
