# Execution Log — Shell Compressor Analytics

- [16:20] 📊 Dataset chosen: Shell Compressor Analytics (AIHack 2020), low-pressure gas compressor SCADA.
- [16:20] 📊 Formulation decision (user): **Both** — classification + anomaly detection. Horizon 4h (24 steps). Split temporal 80/20.
- [17:30] ✅ Added `load_shell_compressor` + detector to `pdm/benchmarks/loaders.py`; registered in `BENCHMARK_REGISTRY`.
- [17:30] ✅ Added `download_shell_compressor` + `DATASET_INFO` entry to `pdm/benchmarks/download.py`.
- [17:34] ✅ Staged `clean_dataset.csv` (106,710 × 363) into `benchmark_data/shell_compressor/`.
- [17:34] ✅ Ran loader → `data/raw_train.csv` (85,368) + `data/raw_test.csv` (21,342), 362 features. `original_index` dropped (leakage). `machine_failure` positives: train 144 (0.17%) / test 72 (0.34%). `meta.validate()` OK.
- [17:34] 🧪 Anomaly-detection baseline (Isolation Forest, contamination 0.01): **AUROC 0.8662 → quality gate PASSED (≥0.75)**. F1=0 at p99 threshold (sparse anomalies; threshold tuning is a Phase-5 experiment). AUPRC 0.0346.
- [17:35] 🧪 Classification baseline (AutoGluon, medium_quality, 300s, --rebalance): trained but ❌ crashed at `evaluate()`.
- [17:40] ⚠️ Pre-existing repo bug: `pdm/fault_prediction/train.py` `--rebalance` (single-label path) adds a `sample_weight` column to train_df but never passes `sample_weight="sample_weight"` to `TabularPredictor`, so AutoGluon treats it as a feature and `evaluate(test_df)` raises `KeyError: ['sample_weight']`. Not specific to this dataset.
- [17:41] 🔧 Re-running classification baseline WITHOUT `--rebalance` to get an honest baseline number.
- [17:48] 🧪 Classification baseline (AutoGluon, medium_quality, 300s): **test F1 0.0 → quality gate FAILED (<0.50)**. Validation F1 ~0.67 but all top models predict all-negative on the temporal test split (0.17% positives). Motivates Phase 5: proper rebalancing, threshold tuning, rolling-window features.
- [17:48] 📊 Verdict: anomaly detection (AUROC 0.866) is the stronger baseline for this sparse-anomaly data; classification needs experimentation. Both wired in as a first-class benchmark (`shell_compressor`) with auto-detection.

## Phase 5 — Experimentation (2026-10-02)

- [14:50] 🔧 Fixed the `--rebalance` bug in `pdm/fault_prediction/train.py` (pass `sample_weight="sample_weight"` to `TabularPredictor`, exclude it from feature_cols; both single-label and `train_single` paths). Confirmed end-to-end: AutoGluon now logs "Values in column 'sample_weight' used as sample weights instead of predictive features."
- [15:02] 🧪 AD E01 Temporal PCA, window=6: **AUROC 0.876 → beats IF baseline (0.866)**; point-adjust F1 0.043 (recall 6/9 segments, 2111 FP).
- [15:05] 🐛 window=12 OOM'd (SIGKILL): `StandardScaler` upcasts to float64 → 3× ~3GB windowed-matrix copies. 🔧 Fixed `pdm/anomaly_detection/temporal.py` to keep scaled arrays float32.
- [15:06] 🧪 AD E01 Temporal PCA, window=12 (after fix): AUROC 0.738 (loses ranking) but point-adjust F1 **0.241** (103 FP) — far more usable operating point. No single AD model dominates both metrics.
- [15:06] 🧪 Classification C01 `--rebalance` (now working): test F1 0.0; AUROC 0.643, oracle-threshold F1 0.028.
- [15:08] 🐛 `--smote` crashed: `ValueError: Input X contains NaN` (SMOTE can't handle missing SCADA values). 🔧 Fixed `pdm/fault_prediction/train.py` to median-impute before `fit_resample`.
- [15:12] 🧪 Classification C02 `--smote` (after fix): test F1 0.0; AUROC 0.665, oracle-threshold F1 0.060 (marginally best classifier).
- [15:14] 🧪 Classification C03 threshold tuning: deployable threshold (tuned on train tail) collapses to 0.5 → F1 0. The classifier's weak ranking (AUROC ~0.65) cannot be rescued by any threshold.
- [15:15] 📊 **Phase-5 verdict:** unsupervised anomaly detection (AUROC 0.87) decisively out-ranks the supervised classifier (AUROC 0.64–0.67) on this sparse, trip-specific label set. AD is the recommended formulation. Classifier F1=0 is a *ranking* problem (too few, too-specific positives), not just a threshold artifact — unlike the AD models, where threshold tuning does recover a usable point.
