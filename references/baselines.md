# Phase 4: Baselines

Build a baseline for the primary model and, **if the user opted in**, for anomaly detection. Both follow the same folder convention:

```
<model_type>/
├── experiments.md        # Experiment backlog & results tracker
├── baseline/
│   ├── README.md         # Results, quality gate outcome
│   ├── runtime.py        # Feature preparation (standalone, importable)
│   ├── data/             # Processed train.csv, test.csv
│   └── model/            # Trained model artifacts + metrics.json
└── experiments/          # Phase 5 experiments (01_name/, 02_name/, ...)
```

## 4A: Anomaly Detection Baseline

> **Skip this section** if the user declined anomaly detection (Path A: `Anomaly detection: no` in User Decisions; Path B: `anomaly_detection: false` in `dataset_meta.json`).

Unsupervised Isolation Forest trained on normal-only data. Complementary to the supervised model — catches novel failure modes not in training labels.

### Training

The `anomaly_detection/baseline/runtime.py` combines feature preparation and training in one script:

```python
"""Anomaly detection baseline: train Isolation Forest on normal-only data."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pathlib import Path
from pdm.fault_prediction import baseline_engineer_features
import pandas as pd

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent.parent
RAW_TRAIN = PROJECT / "data" / "raw_train.csv"
RAW_TEST = PROJECT / "data" / "raw_test.csv"
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "model"

def prepare_and_train():
    # Feature engineering (same as fault prediction)
    train = baseline_engineer_features(pd.read_csv(RAW_TRAIN))
    test = baseline_engineer_features(pd.read_csv(RAW_TEST))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(DATA_DIR / "train.csv", index=False)
    test.to_csv(DATA_DIR / "test.csv", index=False)

    # Train
    from pdm.anomaly_detection.train_anomaly import train as train_ad
    import argparse
    args = argparse.Namespace(
        train=DATA_DIR / "train.csv", test=DATA_DIR / "test.csv",
        output=MODEL_DIR, contamination=0.05, n_estimators=200, seed=42,
    )
    train_ad(args)

if __name__ == "__main__":
    prepare_and_train()
```

Or use the standalone CLI directly:

```bash
uv run python pdm/anomaly_detection/train_anomaly.py \
    --train ./data/raw_train.csv --test ./data/raw_test.csv \
    --output ./anomaly_detection/baseline/model --contamination 0.05
```

**Contamination parameter** (fraction of anomalies expected in training data):
- If training labels exist and positive rate is known → use that rate
- If no labels → default 5%
- For very clean data → try 0.01–0.02

### Evaluation

```bash
uv run python pdm/anomaly_detection/evaluate_anomaly.py \
    --model-dir ./anomaly_detection/baseline/model --test ./data/raw_test.csv

uv run python pdm/anomaly_detection/inference_anomaly.py \
    --model-dir ./anomaly_detection/baseline/model -n 3
```

### Local Inference (Last Day of Telemetry)

The `anomaly_detection/baseline/inference.py` script runs inference on the last N samples of the test set (simulating the last day), with optional per-feature explanations:

```bash
# Default: last 50 samples, no explanations
uv run python anomaly_detection/baseline/inference.py

# Custom sample count with feature contribution explanations
uv run python anomaly_detection/baseline/inference.py -n 10 --explain

# Custom input file
uv run python anomaly_detection/baseline/inference.py --input ./data/new_data.csv --explain
```

**Template for `anomaly_detection/baseline/inference.py`:**

```python
"""Anomaly detection inference on last day of telemetry."""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np, pandas as pd, joblib


def feature_contributions(model, sample_scaled, feature_names, top_k=10):
    """Approximate per-feature contribution via perturbation."""
    base_score = -model.score_samples(sample_scaled.reshape(1, -1))[0]
    contribs = {}
    for j, fname in enumerate(feature_names):
        perturbed = sample_scaled.copy()
        perturbed[j] = 0.0
        contribs[fname] = float(base_score - (-model.score_samples(perturbed.reshape(1, -1))[0]))
    return sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=None)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent / "model")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--top-features", type=int, default=10)
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args()

    # Load artefacts
    metadata = json.loads((args.model_dir / "metadata.json").read_text())
    model = joblib.load(args.model_dir / "isolation_forest.joblib")
    scaler = joblib.load(args.model_dir / "scaler.joblib")
    threshold = json.loads((args.model_dir / "threshold.json").read_text())["threshold"]
    feature_names = metadata["feature_names"]

    # Load input (last N rows = last day)
    if args.input is None:
        data_dir = args.model_dir.parent / "data"
        args.input = data_dir / "test.csv" if (data_dir / "test.csv").exists() else Path("data/raw_test.csv")
    df = pd.read_csv(args.input)
    n = args.n if args.n else min(len(df), 50)
    df = df.tail(n).reset_index(drop=True)

    # Score
    X = scaler.transform(df[feature_names].fillna(0).values)
    scores = -model.score_samples(X)

    # Print per-sample results with optional explanations
    for i in range(len(df)):
        is_anomaly = scores[i] > threshold
        flag = "🚨 ANOMALY" if is_anomaly else "✓ Normal"
        print(f"Sample {i}: score={scores[i]:.6f}  {flag}")
        if args.explain:
            for fname, delta in feature_contributions(model, X[i], feature_names, args.top_features):
                print(f"    {fname:40s} Δ={delta:+.6f}")

    print(f"\nSummary: {(scores > threshold).sum()}/{len(df)} anomalies")
```

### Inference Output Columns

| Column | Description |
|--------|-------------|
| `anomaly_score` | Isolation Forest anomaly score (higher = more anomalous) |
| `is_anomaly` | Binary flag: 1 if score > threshold |
| `threshold` | Decision threshold (from training) |
| `top_anomalous_features` | Semicolon-separated list of `feature(z=N)` — top features by z-score deviation from baseline |

### Adjusting Contamination

| Symptom | Adjustment |
|---------|------------|
| Too many false positives (>15% anomaly rate on test) | Increase contamination → raises threshold |
| Synthetic detection < 80% | Decrease contamination → lower threshold, check feature quality |
| Score distribution is bimodal | Data has distinct operating regimes — consider per-regime AD |

### Output Artefacts

| File | Content |
|------|---------|
| `isolation_forest.joblib` | Trained sklearn model |
| `scaler.joblib` | Fitted StandardScaler |
| `threshold.json` | Threshold value, strategy, rates |
| `metadata.json` | Feature names, config, stats |
| `baseline_stats.json` | Per-feature distributions |
| `anomaly_scores_test.csv` | Score + flag per test sample |
| `score_distribution.png` | Histogram + threshold line |
| `metrics.json` | All metrics + quality gate |

### Quality Gate

- **With labels**: AUROC ≥ 0.75
- **Without labels**: Synthetic detection rate ≥ 80% (injected spikes/drift/level_shifts)
- **Gate failure is NOT blocking** — log and proceed to 4B

**⚠️ Multi-label datasets:** When computing AUROC for AD evaluation on multi-label datasets, use only labels that represent **physical/sensor-detectable faults** as the positive class. Exclude labels for user-behavior or process-related issues that don't manifest as sensor deviations — including them dilutes AUROC and produces misleadingly low scores.

To filter, identify which labels correspond to conditions the anomaly detector could plausibly catch from telemetry data:
```python
# Keep only labels whose root cause would produce anomalous sensor patterns
physical_labels = [c for c in label_cols if is_sensor_detectable(c)]  # project-specific selection
y_anomaly = test[physical_labels].max(axis=1)  # 1 if any physical fault
```

### Pitfalls

- Don't filter the test set — keep mixed normal + anomaly
- Don't use AD scores as features in fault prediction — keep models independent
- Don't skip evaluation — it produces the quality gate and score_distribution.png

## 4B: Fault Prediction Baseline

Minimal transforms — no domain knowledge. Establishes the supervised model performance floor.

### Baseline `runtime.py` Template

```python
"""Baseline feature engineering: minimal transforms on raw data."""
import pandas as pd
from pdm.fault_prediction import baseline_engineer_features

DROP_COLS = ["device_id", "_observation_date"]  # Adjust per project


def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    return baseline_engineer_features(raw_df, drop_cols=DROP_COLS)


if __name__ == "__main__":
    import os
    train = pd.read_csv("./data/raw_train.csv")
    test = pd.read_csv("./data/raw_test.csv")

    train_feat = engineer_features(train)
    test_feat = engineer_features(test)

    os.makedirs("./fault_prediction/baseline/data", exist_ok=True)
    train_feat.to_csv("./fault_prediction/baseline/data/train.csv", index=False)
    test_feat.to_csv("./fault_prediction/baseline/data/test.csv", index=False)
    print(f"Baseline: train={train_feat.shape}, test={test_feat.shape}")
```

```bash
uv run python fault_prediction/baseline/runtime.py

uv run python pdm/fault_prediction/validate_dataset.py --data ./fault_prediction/baseline/data/train.csv

uv run python pdm/fault_prediction/train.py \
    --train ./fault_prediction/baseline/data/train.csv \
    --test ./fault_prediction/baseline/data/test.csv \
    --output ./fault_prediction/baseline/model \
    --time-limit 120 --presets best

uv run python pdm/fault_prediction/inference.py \
    --model-path ./fault_prediction/baseline/model -n 3
```

Write `<model_type>/baseline/README.md` for each — the results table MUST include F1, Precision, Recall, and Positive Rate per label:

```markdown
| Label | F1 | Precision | Recall | Positive Rate |
|-------|-----|-----------|--------|--------------|
```

Log baseline metrics in `log.md`.

## Quality Gates

| Model | Criterion | Threshold |
|-------|-----------|-----------|
| Anomaly Detection (labels) | AUROC | ≥ 0.75 |
| Anomaly Detection (no labels) | Synthetic detection rate | ≥ 80% |
| Classification | Test F1 | ≥ 0.50 |
| Multi-label | Median test F1 | ≥ 0.50 |
| RUL | Test RMSE | < 50% of RUL range |
| Survival | Concordance index | > 0.60 |
