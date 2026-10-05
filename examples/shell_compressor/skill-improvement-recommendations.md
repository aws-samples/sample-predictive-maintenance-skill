# Predictive-Maintenance Skill — Recommended Updates

**From:** a run of the skill end-to-end on a new, genuinely hard dataset
**Context:** We wired the Shell Compressor Analytics dataset (AIHack 2020 — a
low-pressure gas compressor, 106,710 rows × 362 sensor channels, 9 approximate
trip labels, ~0.17–0.34% positive) into the skill as a first-class benchmark and
drove it through data exploration, both baselines (anomaly detection +
classification), and Phase-5 experimentation. The findings below come from that
run. All supporting evidence — run logs, experiment scripts, figures, and the
written-up example — lives in [`examples/shell_compressor/`](./).

This dataset is a good stress test precisely because it is *unfriendly*: extreme
class imbalance, a handful of distinct failure events, wide multivariate input,
missing values, and a sneaky time-proxy column. That combination exercised code
paths and guidance that a tidy benchmark never touches.

---

## Summary

| # | Recommendation | Type | Severity | Effort |
|---|---|---|---|---|
| **B1** | `--rebalance` crashes at `evaluate()` | Bug (patch available) | High | — (done) |
| **B2** | `--smote` crashes on NaN input | Bug (patch available) | High | — (done) |
| **B3** | `TemporalAnomalyDetector` OOMs on wide series | Bug (patch available) | Med | — (done) |
| **R1** | Add integration tests for the imbalance paths | Robustness | High | Low–Med |
| **R2** | Report threshold-independent metrics for classification | Evaluation | High | Low |
| **R3** | Make the classification quality gate imbalance-aware | Evaluation | High | Low |
| **R4** | Add formulation guidance keyed to label economics | Guidance | High | Low |
| **R5** | AD threshold is tuned on the test labels (leak) | Correctness | Med | Med |
| **R6** | Add an automated leakage check in data exploration | Robustness | Med | Med |
| **R7** | Deliver the threshold selection the docs already promise | Consistency | Med | Med |
| **R8** | Memory pre-flight + SMOTE caveat for wide/sparse data | Polish | Low | Low |

The three bugs (B1–B3) are already patched in this branch; everything from R1
down is a recommendation for the maintainer to consider.

---

## Bugs found and fixed this run

These are on code paths that evidently were never exercised end-to-end — the
levers you reach for *specifically* when data is imbalanced. Patches are in this
branch; please review and adopt (or redo to taste).

### B1 — `--rebalance` crashes the classifier at evaluation
`_train_single_label` / `train_single` added a `sample_weight` column to the
training frame but never told AutoGluon it was a weight, so `TabularPredictor`
treated it as a predictive feature and `evaluate(test_df)` raised
`KeyError: ['sample_weight']` (the test frame has no such column).
- **Where:** [`pdm/fault_prediction/train.py`](../../pdm/fault_prediction/train.py)
- **Fix:** pass `sample_weight="sample_weight"` to `TabularPredictor(...)` (see
  [train.py:95](../../pdm/fault_prediction/train.py#L95)) and exclude it from
  `feature_cols`.
- **Repro:** any binary dataset with `--rebalance`.

### B2 — `--smote` crashes on missing values
`SMOTE.fit_resample` interpolates between neighbours and raises
`ValueError: Input X contains NaN`. Real sensor data almost always has gaps, so
this path fails on contact with reality.
- **Where:** the `--smote` block in [`pdm/fault_prediction/train.py`](../../pdm/fault_prediction/train.py)
- **Fix:** median-impute numeric columns before `fit_resample` (all-NaN columns
  fall back to 0).

### B3 — `TemporalAnomalyDetector` runs out of memory on wide series
`StandardScaler.fit_transform` upcasts to float64, so the flattened sliding-window
matrix (`n_samples × window·n_features`) and its PCA reconstruction become several
simultaneous multi-GB copies. On 85k × 362 at a 12-step window this OOM-killed the
process (no traceback — `SIGKILL` leaves none).
- **Where:** [`pdm/anomaly_detection/temporal.py`](../../pdm/anomaly_detection/temporal.py)
- **Fix:** keep the scaled arrays `float32` (reconstruction error does not need
  double precision). Halves the footprint; also benefits the SMAP/MSL benchmark
  the detector was written for.

---

## Recommended enhancements

### R1 — Add integration tests for the imbalance paths  *(highest leverage)*
B1 and B2 both sat on `--rebalance` / `--smote`, and B3 on the wide-data temporal
path. A small synthetic fixture — an imbalanced, wide, NaN-containing frame — run
through each lever to `evaluate()` would have caught all three before a user did.
- **Suggested:** a `tests/` case that trains the classifier with `--rebalance`
  and with `--smote`, and the temporal detector on a wide frame, asserting each
  completes and produces finite metrics. Fast to run, high regression value.

### R2 — Report threshold-independent metrics for classification
The classification baseline reported **F1 = 0.00**, which conflates two very
different situations: "the ranking is fine but 0.5 is the wrong threshold" vs
"the ranking itself is bad." We could only tell them apart by writing a separate
threshold-sweep (see
[`fault_prediction/run_c03_threshold.py`](fault_prediction/run_c03_threshold.py)),
which showed the classifier's problem was genuinely *ranking* (test AUROC ≈ 0.64),
not thresholding — the opposite of what the anomaly detector showed.
- **Suggested:** emit **AUROC** and **average precision** alongside F1 for every
  classification run, plus a **best-achievable (tuned-threshold) F1**. These are
  three cheap lines on top of the probabilities AutoGluon already produces.
- **Where:** metrics assembly in [`pdm/fault_prediction/train.py`](../../pdm/fault_prediction/train.py).

### R3 — Make the classification quality gate imbalance-aware
`Test F1 ≥ 0.50` at the default 0.5 threshold
([SKILL.md:325](../../SKILL.md), [evaluate path in train.py]) is essentially
unachievable at 0.17% positives and flags a usable ranker as a red ❌. The anomaly
side already does this right (threshold-independent AUROC ≥ 0.75 at
[`evaluate_anomaly.py:89`](../../pdm/anomaly_detection/evaluate_anomaly.py#L89)).
- **Suggested:** when the positive rate is below a threshold (e.g. < 1%), gate on
  AUROC/AP (or tuned-threshold F1) rather than default-threshold F1, and print the
  positive rate in the gate message so the number is interpretable.

### R4 — Add formulation guidance keyed to label economics
The headline finding of this run: with **sparse, trip-specific** positives, the
*unsupervised* anomaly detectors (AUROC 0.87) decisively out-ranked the
*supervised* classifier (AUROC 0.64–0.67) — because "learn normal and flag
deviations" generalizes when "learn the failure" has only a handful of examples to
memorize. The current decision flowchart
([SKILL.md:204](../../SKILL.md)) branches on data *type* only, and the Phase-2
checklist recommends anomaly detection "based on data dimensionality"
([SKILL.md:219](../../SKILL.md)).
- **Suggested:** during data exploration, count the **positive rate** and the
  number of **distinct positive events**, and add a flowchart branch: when
  positives are very sparse / few-event, recommend anomaly detection as the
  primary formulation and classification as a secondary check. (Supporting
  "Both" was a real strength of the skill — this just adds a steer.)

### R5 — The anomaly detector tunes its threshold on the test labels
`TemporalAnomalyDetector.train()` optimizes the decision threshold against
`test_df["label"]`
([temporal.py:127–130](../../pdm/anomaly_detection/temporal.py#L127)). Every
reported operating-point F1 is therefore an *optimistic oracle* number, not a
deployable one. (In our run the "deployable" classifier threshold, chosen on a
train tail, collapsed to 0.5 — exactly the gap this hides.)
- **Suggested:** select the threshold on a held-out validation slice of the
  training/normal data, or — at minimum — label the reported F1 as an oracle
  ceiling in the output and docs. AUROC/AUPRC are unaffected (threshold-free) and
  remain the honest headline.

### R6 — Add an automated leakage check in data exploration
The raw file shipped an `original_index` column that monotonically tracks time;
left in the feature set it lets a model "predict" trips from the clock. We caught
and dropped it by hand in the loader. There is currently no leakage guidance in
[`references/data-exploration.md`](../../references/data-exploration.md).
- **Suggested:** a cheap Phase-2/3 guard that flags any feature with
  near-perfect monotonic correlation to row order, or near-perfect correlation to
  the target, and surfaces it for the user to confirm dropping.

### R7 — Deliver the threshold selection the docs already promise
Phase-2 item 4 ([SKILL.md:218](../../SKILL.md)) says the recall/precision/F1
optimization priority "affects … threshold selection," but the classification
pipeline only ever reports metrics at the default 0.5 threshold — there is no
threshold selection step. Either implement operating-point selection (tie it to
the stated priority) or soften the doc. R2 provides the probabilities needed to
do the former.

### R8 — Lower-priority polish
- **Memory pre-flight for the temporal detector:** estimate
  `n_rows × window × n_features × 4 bytes` and warn (suggest a smaller window or
  feature subsampling) before allocating. The float32 fix (B3) helps but does not
  scale indefinitely.
- **SMOTE caveat for wide, very-low-positive data:** in our run SMOTE ran (after
  B2) but produced F1 0 and only a marginal AUROC gain — interpolating synthetic
  positives in 362-dim space from ~144 reals is dubious. Worth a one-line caution
  in the experimentation guidance.
- **Per-experiment output directories:** our experiment runner initially saved
  each window size's model to the same path and silently overwrote it (caught by a
  recomputed-metric sanity check). If the skill ships experiment-runner guidance,
  recommend unique output paths and a recompute-from-artifacts check.

---

## Evidence from this run

Scores are on the temporal 80/20 test split (72 positives / 21,342 rows); AUROC
values were recomputed from saved model outputs, not transcribed.

**Formulation comparison (test AUROC — higher is better):**

| Model | AUROC |
|---|---|
| Temporal PCA · 1h window (unsupervised) | 0.876 |
| Isolation Forest (unsupervised) | 0.866 |
| Classifier · SMOTE (supervised) | 0.665 |
| Classifier · rebalance (supervised) | 0.643 |

**Why default-threshold F1 is misleading (classifier):** F1 at 0.5 = 0.00; even an
oracle threshold tuned on the test set tops out at F1 ≈ 0.06 — the ranking is the
limitation, not the threshold.

**Anomaly-detection window trade-off:** a 1h window ranks best (AUROC 0.876, usable
F1 0.04); a 2h window ranks worse (0.738) but yields a far more usable alarm
(point-adjust F1 0.24, 103 false positives vs 2,111).

**Supporting files:**
- [`log.md`](log.md) — full execution log with timestamps
- [`fault_prediction/experiments.md`](fault_prediction/experiments.md),
  [`anomaly_detection/experiments.md`](anomaly_detection/experiments.md) — Phase-5 results
- [`fault_prediction/run_c03_threshold.py`](fault_prediction/run_c03_threshold.py) — the
  threshold-sweep that motivates R2
- [`make_eda.py`](make_eda.py), [`make_plots.py`](make_plots.py), [`img/`](img/) — the
  exploratory + result figures
- [`README.md`](README.md), [`blog.md`](blog.md) — the written-up example
