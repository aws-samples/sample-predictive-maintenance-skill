# Shell Compressor Analytics — worked example

An energy-sector predictive-maintenance example wired into this skill as a
first-class benchmark. The data is real 10-minute SCADA from a **low-pressure
gas compressor** released for the AIHack 2020 "Compressor Analytics Challenge"
by Shell.

- **Source:** https://github.com/aihack20/shell_challenge (CC — see repo)
- **Asset:** low-pressure gas compressor (LPC)
- **Shape:** 106,710 rows × 362 sensor features + `original_index`
- **Sensors:** ~110 temperature, ~92 pressure, ~30 flow-rate, ~14 speed, plus
  level/gauging and anonymized "unknown" channels. (No vibration channel.)
- **Labels:** 9 anomaly points near equipment trips/shutdowns
  (`anomaly_indexes.txt`). The authors note the anomaly definition is approximate.

## How it maps onto the skill

`pdm/benchmarks/loaders.py::load_shell_compressor` turns the raw CSV into the
common format. It builds a binary `machine_failure` label by flagging the 24
raw-timeline steps (**4 hours**) before each anomaly index — a fault-imminent
window — and drops `original_index` from the features (it is a time proxy that
would leak the label). The split is **temporal 80/20** (row order preserved).

The same `raw_train.csv` / `raw_test.csv` drive **both** formulations:

| Formulation | Model | How the label is used |
|---|---|---|
| Classification | `fault_prediction/train.py` (AutoGluon) | supervised target |
| Anomaly detection | `anomaly_detection/train_anomaly.py` (Isolation Forest) | excluded from features; normal-only training; AUROC eval |

## Reproduce

```bash
# 1. Download + prepare (writes raw_train/raw_test/dataset_meta into the folder)
uv run python -m pdm.benchmarks.download ./benchmark_data shell_compressor

# 2. Point both baselines at the prepared CSVs (here under examples/shell_compressor/data)
uv run python pdm/anomaly_detection/train_anomaly.py \
    --train data/raw_train.csv --test data/raw_test.csv \
    --output anomaly_detection/baseline/model --contamination 0.01
uv run python pdm/anomaly_detection/evaluate_anomaly.py \
    --model-dir anomaly_detection/baseline/model --test data/raw_test.csv

uv run python pdm/fault_prediction/train.py \
    --train data/raw_train.csv --test data/raw_test.csv \
    --output fault_prediction/baseline/model \
    --presets medium_quality --time-limit 300 --rebalance
```

## Baseline results

| Model | Metric | Value | Gate |
|---|---|---|---|
| Anomaly detection (Isolation Forest, contamination 0.01) | AUROC | 0.866 | ✅ ≥ 0.75 |
| Classification (AutoGluon, medium_quality, 300s) | F1 | 0.00 | ❌ ≥ 0.50 |

Class imbalance is severe (~0.17% train / ~0.34% test positive):
- **Anomaly detection is the stronger fit.** Judge it by **AUROC**
  (threshold-independent); its F1 at the default p99 threshold is ~0 until the
  threshold is tuned (a Phase-5 experiment).
- **The classifier collapses to all-negative** on the temporal test split at the
  default threshold (test F1 0.0, though validation F1 ≈ 0.67). This is the
  expected baseline for such sparse labels and motivates Phase-5 experimentation:
  rebalancing, decision-threshold tuning, and rolling-window features.

## Phase-5 experiments

Full detail in [`anomaly_detection/experiments.md`](anomaly_detection/experiments.md)
and [`fault_prediction/experiments.md`](fault_prediction/experiments.md).

**Anomaly detection** — temporal PCA reconstruction vs. the IF baseline:

| Model | AUROC | Point-adjust F1 (oracle threshold) |
|---|---|---|
| Isolation Forest (baseline) | 0.866 | 0.00 |
| Temporal PCA, window=6 (1h) | **0.876** | 0.043 |
| Temporal PCA, window=12 (2h) | 0.738 | **0.241** |

**Classification** — rebalancing / SMOTE / threshold tuning:

| Experiment | Default F1 | Test AUROC | Oracle-threshold F1 |
|---|---|---|---|
| `--rebalance` | 0.00 | 0.643 | 0.028 |
| `--smote` | 0.00 | 0.665 | 0.060 |

**Takeaway:** unsupervised anomaly detection (AUROC ≈ 0.87) decisively out-ranks
the supervised classifier (AUROC ≈ 0.64–0.67) on this sparse, trip-specific label
set — AD is the recommended formulation. The classifier's F1=0 is a *ranking*
limitation (144 trip-specific positives don't generalise), not just a threshold
artifact; for the AD models, by contrast, threshold tuning does recover a usable
operating point.

> Two pre-existing bugs in `pdm/fault_prediction/train.py` surfaced and were fixed
> during these experiments: `--rebalance` crashed at `evaluate()` (the
> `sample_weight` column was not registered with AutoGluon), and `--smote` crashed
> on NaNs (no imputation before oversampling). A float64→float32 memory fix was
> also made in `pdm/anomaly_detection/temporal.py` so the windowed detector fits in
> ~8GB. The original baseline table above was produced before these fixes.

See `log.md` for the full run log.
