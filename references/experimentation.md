# Experimentation Loop

Iteratively improve model performance through hypothesis-driven experiments. The baseline is already complete from Phase 4 — this phase builds on it. Applies to **both** anomaly detection and fault prediction models following the same conventions.

## ⚠️ Gotchas (Feature Engineering & Training)

- **Zero-variance features**: Always drop zero-variance features in `runtime.py` before saving — EAV pivots commonly produce them from sensors that never vary.
- **`encode_categoricals` column invalidation**: `encode_categoricals(df, cols)` drops original columns and creates `*_enc` replacements. Always recompute column lists after calling it — reusing a list computed before the call will raise `KeyError`.
- **Multi-label label naming**: All label columns MUST be prefixed with `label_` for auto-detection by `validate_dataset.py` and `train.py`.

## Step 0: Build Domain Knowledge Base (LLM Wiki)

**Before proposing any experiments**, build a focused LLM Wiki covering the specific predictive maintenance formulation under development. This ensures experiments are grounded in the state of the art rather than ad-hoc intuition.

**Use the knowledge-acquisition skill** (located in `.kiro/skills/knowledge-acquisition/SKILL.md`) to research and build a `./wiki/` for the project. Follow that skill's full workflow (gather sources → distill → write wiki pages).

**Research queries** — tailor to the formulation:
- RUL → "remaining useful life prediction", "degradation modeling"
- Classification → "predictive maintenance classification", "failure prediction IoT"
- Multi-label → "multi-label fault diagnosis", "multi-fault classification"
- Survival → "survival analysis predictive maintenance", "Cox proportional hazards equipment"
- Anomaly Detection → "unsupervised anomaly detection IoT", "isolation forest condition monitoring"

**Required wiki pages** (create under `./wiki/`):
- `concepts/feature-engineering-pdm.md` — state-of-the-art feature engineering for this formulation
- `concepts/<formulation>-techniques.md` — best models, loss functions, evaluation metrics
- `concepts/domain-features.md` — domain-specific features for the equipment type

Each page must end with a `## Implications for Experiments` section listing concrete, actionable hypotheses derived from the literature.

**Skip condition**: If `./wiki/concepts/feature-engineering-pdm.md` already exists and is relevant to the current formulation, skip this step.

---

## Step 0B: Propose & Prioritize Experiments

**Before running any experiments**, write `./<model_type>/experiments.md` with a prioritized backlog. The baseline is already in the Executed section. Propose 3–5 follow-up experiments ranked by expected impact.

```markdown
# Experiments

## Planned

Prioritized by expected impact (highest first). Re-prioritize after each experiment completes.

### 1. [Experiment name] (01_name)
- **Hypothesis**: [Why this should improve metrics over baseline]
- **Expected impact**: [Which labels/metrics should improve and why]
- **Priority rationale**: [Why this is ranked here vs. lower experiments]

### 2. [Next experiment name] (02_name)
- **Hypothesis**: [...]
- **Expected impact**: [...]
- **Priority rationale**: [...]

### 3. ...

## Executed

### baseline — ✅ [Result]
- **Result**: Median F1 = 0.XX, Mean F1 = 0.XX
- **Key learnings**: [Top features, weak labels, what the model struggles with]
- **Decision**: Performance floor established

## Final Retraining

(Added after all experiments complete and improvements are combined. Use `pdm.training_time.format_retraining_options()` to generate with accurate time estimates.)

### Final Model Retraining

The combined model merges improvements from all experiments that beat the baseline. Choose the final training configuration:

| Option | Configuration | Estimated Time | When to use |
|--------|--------------|----------------|-------------|
| A) **Skip** | Use combined model as-is | 0 min | Results already meet requirements; no further iteration needed |
| B) **Extended** (recommended) | `--time-limit 120 --presets good --skip-importance` | ~[X] min | Rebuild with combined features and more model iterations |
| C) **Full** | `--time-limit 300 --presets best_quality --skip-importance` | ~[Y] min | Only for datasets >50K rows where stacking reliably helps |

[Answer]:
```

**Prioritization criteria** (highest to lowest):
1. Literature-backed techniques from the LLM Wiki that directly address weak labels or known failure modes
2. Features that directly address weak labels identified in the baseline (informed by wiki domain knowledge)
3. Domain-informed interactions grounded in physical reasoning (CAAFE-style semantic reasoning, supported by wiki references)
4. Signal representation changes (rolling windows, frequency features, trend indicators — as documented in wiki)
5. Feature selection / dimensionality reduction
6. Hyperparameter or ensemble variations

**CAAFE-style semantic reasoning**: For each proposed experiment, reason about column semantics:
- What do the top features physically represent?
- What interactions would be physically meaningful? (e.g., voltage_range = max - min indicates instability)
- Which sensors are causally upstream of each failure mode?

## Step 1: User Review — STOP AND WAIT

After writing the `## Planned` section in `<model_type>/experiments.md`, present the plan to the user for approval.

**For small jobs (≤5 experiments, dataset <500MB, estimated time <60 min locally):**
Default to local execution. Do NOT ask infrastructure questions — just ask for plan approval:

```markdown
## Approval

Experiments will run **locally** (estimated ~[X] min total).

A) **Approve** — run experiments in the proposed order
B) **Edit** — I'll modify experiments.md and tell you to continue
C) **Skip** — use the baseline as final model

[Answer]:
```

**For large jobs (>5 experiments, dataset >500MB, or estimated time >60 min):**
Ask the full infrastructure questions (local vs SageMaker, spot vs on-demand) in addition to approval.

```markdown
## Infrastructure Questions

### Question 1
**Where should the experiments run?**

| Option | Time estimate | Cost |
|--------|--------------|------|
| A) **Local** (sequential) | ~[N] experiments × 5 min = [X] min total | Free |
| B) **SageMaker** (parallel) | ~5 min setup + 5 min training = [Y] min total | ~$0.05–0.15/experiment |

[Answer]:

### Question 2
**If remote — spot or on-demand instances?**

(Skip if Question 1 = Local)

[Answer]:

### Question 3
**Approve the experiment plan?**

A) **Approve** — run experiments in the proposed order
B) **Edit** — I'll modify experiments.md and tell you to continue
C) **Skip** — use the baseline as final model

[Answer]:
```

**Time estimate calculation**: Use `pdm.training_time.estimate_training_minutes()` for accurate estimates. For experiments (fast iteration), use `--presets good --time-limit 60`. See the "Training Time Estimation" section below for the correct formula — the old "N labels × 60s" heuristic significantly underestimates multi-label training time.

**Present the questions to the user and STOP.** Do NOT proceed until `[Answer]:` fields are filled.

After the user responds, **write the answers into `experiments.md`** and log the decision:
```bash
echo "- [$(date +%H:%M)] 📊 Decision: Experiments will run [locally|on SageMaker (spot|on-demand)]" >> log.md
```

If the user chooses Question 3 = C (Skip), proceed directly to Phase 6 (Documentation).
If the user chooses Question 3 = B (Edit), re-read experiments.md after they notify and follow the new plan.

## Logging Mandate

**Update `log.md` at EVERY step** — not just at the end of a phase. Log:
- Start of each experiment (hypothesis in one line)
- Training completion (metrics summary)
- Decisions (keep/discard, next experiment chosen)
- Any errors or unexpected findings

## Workflow

```
┌─────────────────────────────────────────────────────────┐
│  0) PROPOSE  — Write experiments.md with ranked backlog  │
│  1) REVIEW   — Present to user, wait for approve/edit/   │
│               skip                                       │
│                                                          │
│  For each experiment (starting from 01):                 │
│                                                          │
│  A) HYPOTHESIZE — Why might this improve performance?    │
│  B) IMPLEMENT   — Write runtime.py (raw → model input)  │
│  C) TRAIN       — Run AutoGluon on processed features    │
│  D) LOG         — Update log.md + experiments.md         │
│                                                          │
│  Stop when: quality gate passes AND no obvious           │
│  improvement hypotheses remain, OR max 5 experiments.    │
│                                                          │
│  COMBINE & BUILD FINAL:                                  │
│    1. Identify all experiments that beat baseline         │
│    2. Merge their improvements into a single runtime.py  │
│    3. Train the combined model                           │
│    4. Verify combined ≥ best individual experiment       │
│    5. Ask user about final retraining:                   │
│       A) Skip — use combined model as-is                 │
│       B) Extended (recommended) — --time-limit 120       │
│          --presets good --skip-importance                 │
│       C) Full — --time-limit 300 --presets best_quality  │
│          --skip-importance (only if >50K rows)           │
│  (include time estimates from pdm.training_time)         │
└─────────────────────────────────────────────────────────┘
```

### ⏱️ Time Budget Rule

**Each experiment must complete training in under 30 minutes** (excluding container setup and data upload time). Use `--presets good --time-limit 60` for multi-label (60s per label × N labels). For RUL/classification, `--time-limit 180 --presets good`.

The purpose of experiments is **fast iteration over a large search space** — not squeezing maximum performance from each attempt. The combined model merges all improvements for deployment. Final retraining is optional and presented as a user choice (see "Combining Improvements into the Final Model" below).

## Experiment Folder Structure

Each experiment lives in `./<model_type>/experiments/NN_name/`:

```
<model_type>/                 # anomaly_detection/ or fault_prediction/
├── experiments.md            # Backlog & results tracker
├── baseline/                 # Already complete from Phase 4
│   ├── README.md
│   ├── runtime.py
│   ├── data/
│   └── model/
└── experiments/
    ├── 01_<descriptive_name>/
    │   ├── README.md       # Hypothesis, approach, results, comparison
    │   ├── runtime.py      # raw data → model-ready features (standalone)
    │   ├── data/
    │   │   ├── train.csv   # Processed features for training
    │   │   └── test.csv    # Processed features for testing
    │   └── model/
    │       ├── metrics.json
    │       ├── metadata.json
    │       └── ...         # Model-specific artifacts
    ├── 02_<descriptive_name>/
    │   └── ...
    └── ...
```

## Experiments 01+: Hypothesis-Driven Improvement

### A) Hypothesize

Before implementing, state a clear hypothesis based on **domain knowledge from the wiki**:

1. **Check the LLM Wiki** — read `./wiki/concepts/feature-engineering-pdm.md`, `./wiki/concepts/<formulation>-techniques.md`, and any domain-specific pages. Extract experiment ideas from the `## Implications for Experiments` sections.
2. **Analyze previous experiment** — which features had highest/lowest importance? Which labels underperformed?
3. **Form hypothesis** — "Adding [specific feature/transform] should improve [metric] because [domain reasoning from wiki page X]"

**Hypothesis sources (priority order):**
1. Literature-driven ideas from the LLM Wiki (techniques proven in peer-reviewed work for this formulation)
2. Feature importance from previous experiment (low-importance features → replace; high-importance → derive interactions)
3. CAAFE-style semantic reasoning: read column descriptions → propose meaningful interactions grounded in wiki knowledge
4. Domain-specific degradation physics documented in the wiki (e.g., thermal cycling fatigue, bearing wear progression)

### B) Implement

Write `runtime.py` for the new experiment. **Each runtime.py must be standalone** — it reads from `./data/raw_train.csv` and `./data/raw_test.csv` and produces its own processed data.

```python
"""Experiment 01: Thermal interaction features.

Hypothesis: Temperature differentials (process_temp - ambient_temp) indicate
thermal stress that precedes overheating failures. Adding delta features
should improve label_Overheating F1.
"""
import pandas as pd
from pdm.data.utils import encode_categoricals

def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    # ... all baseline transforms ...

    # NEW: Thermal interaction features
    if "process_temp_mean" in df.columns and "ambient_temp_mean" in df.columns:
        df["thermal_delta_mean"] = df["process_temp_mean"] - df["ambient_temp_mean"]
        df["thermal_stress_ratio"] = df["process_temp_max"] / (df["ambient_temp_mean"] + 1)

    df = df.fillna(0)
    return df
```

**Rules for runtime.py:**
- MUST define `engineer_features(raw_df) -> pd.DataFrame` — pure, deterministic transforms only
- `engineer_features` must NOT contain data-dependent logic (no variance checks, no importance-based selection, nothing that requires seeing multiple samples). It must produce the same columns for any input.
- Feature selection is handled by `train.py` automatically — do NOT call `drop_zero_variance` in runtime.py
- At inference: call `engineer_features()` then `align_to_model()` (which reads the column list from `metadata.json`)
- MUST be importable (no side effects at import time)
- MUST include the hypothesis as a module docstring
- MUST handle missing columns gracefully (some raw data may lack expected columns)
- Previous experiment's transforms can be included and extended

### C) Train & Verify

**Fault Prediction:**
```bash
uv run python pdm/fault_prediction/train.py \
    --train ./fault_prediction/experiments/NN_name/data/train.csv \
    --test ./fault_prediction/experiments/NN_name/data/test.csv \
    --output ./fault_prediction/experiments/NN_name/model \
    --time-limit 60 --presets good

uv run python pdm/fault_prediction/inference.py \
    --model-path ./fault_prediction/experiments/NN_name/model -n 3
```

**Anomaly Detection:**
```bash
uv run python pdm/anomaly_detection/train_anomaly.py \
    --train ./anomaly_detection/experiments/NN_name/data/train.csv \
    --test ./anomaly_detection/experiments/NN_name/data/test.csv \
    --output ./anomaly_detection/experiments/NN_name/model \
    --contamination 0.05

uv run python pdm/anomaly_detection/evaluate_anomaly.py \
    --model-dir ./anomaly_detection/experiments/NN_name/model \
    --test ./anomaly_detection/experiments/NN_name/data/test.csv

uv run python pdm/anomaly_detection/inference_anomaly.py \
    --model-dir ./anomaly_detection/experiments/NN_name/model -n 3
```

Use `--time-limit 60 --presets good` for fast iteration during experimentation. The goal is quick feedback to explore a larger search space — each experiment should complete within 30 minutes (excluding setup time). After all experiments finish, improvements are combined into a final model (see "Combining Improvements into the Final Model").

The inference step confirms the model loads correctly and produces predictions on held-out samples. If it fails: check that `metadata.json` exists (auto-created by training scripts) and that feature names match.

### D) Log

Create `./<model_type>/experiments/NN_name/README.md`:

```markdown
# Experiment NN: Name

## Hypothesis
[Why this should improve performance]

## Changes from Previous
- Added: [new features]
- Removed: [dropped features]
- Modified: [changed transforms]

## Results
| Metric | Previous Best | This Experiment | Delta |
|--------|--------------|-----------------|-------|
| Mean F1 | 0.XX | 0.XX | +/-0.XX |
| Median F1 | 0.XX | 0.XX | +/-0.XX |

### Per-Label Breakdown (if multi-label)
| Label | Previous F1 | New F1 | Delta | Precision | Recall |
|-------|-------------|--------|-------|-----------|--------|

### Top Features
[From metrics.json feature importance]

![Feature Importance](./model/shap_summary.png)

## Conclusion
[Keep/discard? What did we learn? Next hypothesis?]
```

**After every experiment completes**:
1. Move the experiment from `## Planned` to `## Executed` with results
2. Re-prioritize remaining planned experiments based on what was learned
3. Add new experiment ideas if the results suggest them
4. Update `log.md` with the experiment outcome

## Stopping Criteria

Stop the experimentation loop when:
1. **Quality gate passes** AND
2. **No obvious improvement hypotheses remain** (last experiment showed <2% improvement), OR
3. **Maximum 5 experiments** reached (diminishing returns)

### Quality Gates

| Model | Criterion | Threshold |
|-------|-----------|-----------|
| Anomaly Detection (labels) | AUROC | ≥ 0.75 |
| Anomaly Detection (no labels) | Synthetic detection rate | ≥ 80% |
| Classification | Test F1 | ≥ 0.50 |
| Multi-label | Median test F1 | ≥ 0.50 |
| RUL | Test RMSE | < 50% of RUL range |
| Survival | Concordance index | > 0.60 |

## Validation Gate

After the combined model is built, run validation on its processed data (fault prediction only):

```bash
uv run python pdm/fault_prediction/validate_dataset.py --data ./fault_prediction/experiments/combined/data/train.csv
```

Checks: NaN in target, non-numeric columns, zero-variance features, ID leakage, class balance.

## Training Time Estimation

### Why the Naive Estimate Is Wrong

The old formula "N labels × 60s + overhead ≈ 5 min" is **dangerously inaccurate** for multi-label classifiers with larger time budgets. Two compounding factors:

**1. Sequential label training**: AutoGluon trains each label as an independent TabularPredictor **sequentially**:
- `--time-limit 300` with 9 labels = up to **9 × 300s = 45 min** of pure training
- `--presets best_quality` uses the **full budget** (stacking, bagging, repeated k-fold) — it won't finish early

**2. Feature importance (the hidden cost)**: After training each label, `train.py` calls `predictor.feature_importance()` which does permutation importance:
- Shuffles each feature × 5 times (default `num_shuffle_sets=5`) and calls `predict_proba` on 5000 rows each time
- Total calls per label = `n_features × 5` = **2,215 prediction passes** (for 443 features)
- With `best_quality`, the model is a stacked WeightedEnsemble_L2 that invokes ALL base models per call
- **Feature importance alone can take longer than training** — ~67 min for 9 labels × 443 features with best_quality
- For comparison, with `good` presets (single LightGBM winner), feature importance is only ~10 min total

### Correct Formula

```
total_minutes = training + feature_importance + overhead + sagemaker_setup

training       = (time_limit × n_labels × data_factor × preset_factor) / 60
feature_importance = (n_features × num_shuffle_sets × predict_cost × n_labels) / 60
overhead       = data loading + metric computation (~0.5–2 min)
sagemaker_setup = ~6 min (provisioning + pip install)
```

Where:
- **data_factor** (0.5–1.0): Small datasets finish early; large datasets use full budget
- **preset_factor**: `good` ≈ 0.7, `best_quality` ≈ 1.0 of the time_limit
- **num_shuffle_sets**: 5 (AutoGluon default)
- **predict_cost**: ~0.03s/call for `good` (single model), ~0.20s/call for `best_quality` (stacked ensemble)

### Use the Utility Function

Rather than computing this manually, use `pdm.training_time`:

```python
from pdm.training_time import estimate_training_minutes, format_retraining_options

# Estimate a specific configuration
est = estimate_training_minutes(
    formulation="multilabel",
    n_labels=9,
    time_limit=300,
    presets="best_quality",
    n_train_rows=20000,
    n_features=460,
    execution="local",
)
print(est["total_minutes"])  # e.g. 46.2
print(est["breakdown"])       # per-phase breakdown

# Generate the retraining questionnaire options
options_md = format_retraining_options(
    n_labels=9,
    n_train_rows=20000,
    n_features=460,
    formulation="multilabel",
    execution="local",
)
```

### Reference Times (9 labels, 20K rows, 460 features, local execution)

| Configuration | Training | FI | Total | With `--skip-importance` |
|--------------|----------|-----|-------|--------------------------|
| `--time-limit 60 --presets good` | ~6 min | ~10 min | ~17 min | **~7 min** |
| `--time-limit 120 --presets good` | ~12 min | ~10 min | ~23 min | **~13 min** |
| `--time-limit 300 --presets best_quality` | ~44 min | ~67 min | ~112 min | **~45 min** |

**Key insight**: `--skip-importance` saves 40-60% of total time. Feature importance is useful during experiments (to inform next hypothesis) but adds no value for the final deployed model. **Always use `--skip-importance` for final retraining.**

## Combining Improvements into the Final Model

After all experiments complete, merge the improvements from every experiment that beat the baseline into a single combined model. This captures complementary gains that individual experiments discovered independently.

### Step 1: Identify Winning Experiments

Compare metrics across all experiments (including baseline). An experiment "wins" if it improved the primary metric (median F1, RMSE, C-index) over the baseline by ≥1%.

```markdown
| Experiment | Primary Metric | Delta vs Baseline | Status |
|------------|---------------|-------------------|--------|
| baseline   | 0.XX          | —                 | floor  |
| 01_name    | 0.XX          | +0.XX             | ✅ merge |
| 02_name    | 0.XX          | -0.XX             | ❌ skip |
| 03_name    | 0.XX          | +0.XX             | ✅ merge |
```

If only one experiment beat baseline, use it directly (no merge needed) — skip to Step 4.

### Step 2: Merge Improvements into a Combined `runtime.py`

Create `./<model_type>/experiments/combined/runtime.py` that includes the feature engineering from **all winning experiments**:

1. Start from the baseline `runtime.py`
2. Add the new features/transforms from each winning experiment
3. If two experiments modified the same feature differently, keep the version from the higher-performing experiment
4. Document which experiment contributed each block with inline comments

```python
"""Combined: merges improvements from experiments [01, 03, ...].

Includes:
- From 01_interactions: thermal delta features, voltage range
- From 03_rolling: 7-day rolling aggregates on vibration sensors
"""
```

**Rules for merging:**
- Only include transforms that contributed to the metric improvement (check feature importance from each experiment)
- Drop features that had near-zero importance in their source experiment
- If experiments conflict (e.g., one adds a feature, another removes it), prefer the experiment with higher metric gain

### Step 3: Train & Verify the Combined Model

```bash
# Generate combined data
uv run python -c "
from <model_type>.experiments.combined.runtime import engineer_features
import pandas as pd
raw_train = pd.read_csv('./data/raw_train.csv')
raw_test = pd.read_csv('./data/raw_test.csv')
engineer_features(raw_train).to_csv('./<model_type>/experiments/combined/data/train.csv', index=False)
engineer_features(raw_test).to_csv('./<model_type>/experiments/combined/data/test.csv', index=False)
"

# Train
uv run python pdm/fault_prediction/train.py \
    --train ./<model_type>/experiments/combined/data/train.csv \
    --test ./<model_type>/experiments/combined/data/test.csv \
    --output ./<model_type>/experiments/combined/model \
    --time-limit 60 --presets good

# Verify inference
uv run python pdm/fault_prediction/inference.py \
    --model-path ./<model_type>/experiments/combined/model -n 3
```

### Step 4: Validate Combined ≥ Best Individual

Compare the combined model's primary metric against the best individual experiment:
- If combined ≥ best individual → use combined as the final model
- If combined < best individual by >2% → the merged features interact poorly; fall back to the single best experiment and log the reason
- If combined is within 2% of best individual → prefer combined (more robust feature set)

### Step 5: Ask About Final Retraining

Present the retraining question in `<model_type>/experiments.md` (see "Final Retraining Question" below).

### Why Extended Rebuild Is Recommended

The combined `runtime.py` merges feature engineering from multiple experiments, but during experimentation each was trained with a low time budget (`--time-limit 60 --presets good`). Rebuilding with `--time-limit 120` gives AutoGluon more iterations to find optimal interactions across the merged feature set. This consistently outperforms the quick-iteration models from the experiment phase.

**When to skip the rebuild**:
- The combined model already far exceeds the quality gate (e.g., >0.85 F1) and time is constrained
- Only one experiment beat the baseline (no merge happened, so the model was already trained)

**When Full is worthwhile**:
- Dataset >50K rows (stacking generalizes better with more data)
- Single-label classification or RUL regression (no N× multiplier)

### Final Retraining Question

After building the combined model, present this question in `<model_type>/experiments.md`:

```python
from pdm.training_time import format_retraining_options

# Use actual project values
print(format_retraining_options(
    n_labels=N,           # from metadata.json or label count
    n_train_rows=ROWS,    # from train.csv
    n_features=FEATURES,  # from combined train.csv columns
    formulation="multilabel",
    execution="local",    # or "sagemaker" based on earlier decision
))
```

This generates a table like:

```markdown
### Final Model Retraining

The combined model merges improvements from all experiments that beat the baseline. Choose the final training configuration:

| Option | Configuration | Estimated Time | When to use |
|--------|--------------|----------------|-------------|
| A) **Skip** | Use combined model as-is | 0 min | Results already meet requirements; no further iteration needed |
| B) **Extended** (recommended) | `--time-limit 120 --presets good --skip-importance` | ~13 min | Rebuild with combined features and more model iterations |
| C) **Full** | `--time-limit 300 --presets best_quality --skip-importance` | ~45 min | Only for datasets >50K rows where stacking reliably helps |

> ⚠️ **Warning**: With <50K training rows, `best_quality` often overfits (stacked ensembles hurt). Option B is recommended over C.

[Answer]:
```

**Present this question and STOP.** Do NOT proceed with retraining until the user answers. If the user picks A (Skip), proceed directly to documentation/deployment.

## Error Analysis & Interpretability — Validate the Combined Model

After building the combined model, analyze failures and verify feature importance before deploying. This step catches systematic issues and builds domain expert trust.

### 1. Error Analysis — Where Does the Model Fail?

```python
import pandas as pd, json

test = pd.read_csv("./data/test.csv")
with open("./model/metrics.json") as f:
    metrics = json.load(f)

# Load predictions (re-run inference if needed)
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor.load("./model/ag_model")
preds = predictor.predict(test.drop(columns=[c for c in test.columns if c.startswith("label_")]))

# False negatives (missed failures) — most critical
fn_mask = (test["label_X"] == 1) & (preds == 0)
fn_samples = test[fn_mask]

# Check for patterns in missed failures:
print(f"False negatives: {fn_mask.sum()} / {test['label_X'].sum()} failures missed")
print(fn_samples[["device_family_enc", "device_age_days", "n_links"]].describe())
```

**What to look for:**
- Are failures missed for a specific device subgroup (e.g., one hardware revision)?
- Are failures missed at a specific time range (e.g., early in lookback window)?
- Is there a feature range where the model always gets it wrong?

**If patterns emerge:**
- Subgroup-specific failures → add interaction features or train per-subgroup models (new experiment)
- Time-dependent misses → lookback window may be too short
- Random misses with no pattern → model is at its ceiling for this data

### 2. Feature Importance — Is the Model Reasoning Correctly?

AutoGluon's `train.py` saves permutation-based importance to `metrics.json` automatically:

```python
# Single-label
if "feature_importance" in metrics:
    for feat, imp in list(metrics["feature_importance"].items())[:10]:
        print(f"  {feat:40s} {imp:.4f}")

# Multi-label
if "labels" in metrics:
    for label, info in metrics["labels"].items():
        if "top_features" in info:
            print(f"{label}: {info['top_features']}")
```

Generate importance plot:
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

importance = predictor.feature_importance(test, silent=True)
fig, ax = plt.subplots(figsize=(10, 6))
importance.head(15)["importance"].plot.barh(ax=ax)
ax.set_xlabel("Importance (permutation-based metric drop)")
plt.tight_layout()
plt.savefig("./model/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
```

**Validation checklist:**
1. Top features are physically meaningful (vibration, temperature, memory usage — not IDs/timestamps)
2. No single feature dominates >50% without justification → suspect data leakage
3. Feature direction is reasonable (rising temperature → thermal failure makes sense)

### 3. Decision

| Finding | Action |
|---------|--------|
| Systematic failures in a device subgroup | New experiment: add subgroup-aware features or stratify |
| Top feature is an ID or timestamp | Data leakage — fix `runtime.py`, re-run |
| One feature >50% importance, no physical basis | Leakage or proxy variable — investigate and remove |
| Failures are random, features are sensible | Model is at ceiling — accept and deploy |
| Recall too low on critical labels | Adjust threshold or add cost-sensitive weighting (new experiment) |

If action is "accept and deploy" → log and proceed to Phase 6.
If action requires another experiment → add it to the backlog, run it, rebuild the combined model.

Log in `log.md`:
```bash
echo "- [$(date +%H:%M)] 📊 Error analysis: FN pattern=[X], top features=[Y,Z] — [accept/new experiment]" >> log.md
```

## Deploying the Final Model

The combined experiment folder (or single best experiment if no merge was needed) is **self-contained for deployment**:
- `runtime.py` contains the complete feature preparation pipeline (raw → model input)
- `model/` contains all trained model artifacts
- `model/metadata.json` documents feature names, formulation, training date

To run inference on new raw data (fault prediction example):
```python
from fault_prediction.experiments.combined.runtime import engineer_features
import pandas as pd
from autogluon.tabular import TabularPredictor

raw = pd.read_csv("new_raw_data.csv")
features = engineer_features(raw)
predictor = TabularPredictor.load("./fault_prediction/experiments/combined/model/ag_model/label_X")
predictions = predictor.predict(features)
```

## Error Recovery

| Failure | Recovery |
|---------|----------|
| Validation fails | Fix runtime.py per `validate_dataset.py` output, regenerate |
| AutoGluon error | Run `validate_dataset.py` — check NaN, non-numeric cols, zero-variance |
| All experiments fail quality gate | Revisit Phase 2 decisions (label, horizon, formulation) |

## Remote Mode: Parallel Experiments on SageMaker

When running multiple experiments, you can dispatch them in parallel on SageMaker instead of waiting sequentially. This turns 5×5min = 25min into 5min wallclock.

### When to Use Remote Mode

- More than 2 experiments to run
- Large datasets (>50K training rows) where local training is slow
- User wants results faster and has AWS credentials configured

### Setup — Multi-label (pre-processed data per experiment)

Each experiment has its own runtime.py that produces separate train/test CSVs. Create an `experiments.json`:

```json
[
    {"name": "01_interactions", "train": "experiments/01/data/train.csv", "test": "experiments/01/data/test.csv", "time-limit": 60, "presets": "good"},
    {"name": "02_cross_signals", "train": "experiments/02/data/train.csv", "test": "experiments/02/data/test.csv", "time-limit": 60, "presets": "good"},
    {"name": "03_feature_selection", "train": "experiments/03/data/train.csv", "test": "experiments/03/data/test.csv", "time-limit": 60, "presets": "good"}
]
```

### Setup — RUL/Classification (shared data, different hyperparameters)

All experiments share the same raw data but vary hyperparameters:

```json
[
    {"name": "baseline", "window-size": 30, "backend": "optuna", "n-trials": 100},
    {"name": "window20", "window-size": 20, "backend": "optuna", "n-trials": 100},
    {"name": "autogluon_w30", "window-size": 30, "backend": "autogluon", "time-limit": 300}
]
```

### Submit & Monitor

```bash
# Multi-label experiments (uploads per-experiment data):
uv run python pdm/remote/parallel.py \
    --experiments experiments.json --formulation multilabel \
    --instance-type ml.m5.2xlarge

# RUL experiments (shared raw data):
uv run python pdm/remote/parallel.py \
    --train data/raw_train.csv --test data/raw_test.csv \
    --experiments experiments.json --formulation rul \
    --instance-type ml.m5.4xlarge
```

After submission, the script automatically starts a **live progress dashboard** showing:
- Per-label completion (F1/precision/recall as each label finishes training)
- Spot interruption count and restarts
- ETA based on average label training time
- Final comparison table when all jobs complete

### Monitor Existing Jobs

```bash
# One-shot status check:
uv run python pdm/remote/parallel.py --monitor

# Live watch (refreshes every 30s until all complete):
uv run python pdm/remote/parallel.py --monitor --watch

# With CloudWatch log streaming:
uv run python pdm/remote/parallel.py --monitor --logs --watch
```

### Single Job (for debugging)

```bash
uv run python pdm/remote/submit.py \
    --train data/raw_train.csv --test data/raw_test.csv \
    --formulation rul --optuna --n-trials 100 --window-size 30
```

### Instance Type Selection

| Workload | Instance | Cost/hr | Notes |
|---|---|---|---|
| Multi-label (AutoGluon, 6 labels) | `ml.m5.2xlarge` | ~$0.46 | 8 vCPU — AG manages own parallelism |
| Optuna HPO (RF/XGB/GBR) | `ml.m5.4xlarge` | ~$0.92 | 16 vCPU — Optuna trials parallelize via n_jobs=-1 |
| Large datasets (>100K rows) | `ml.m5.4xlarge` | ~$0.92 | More memory for windowing |

### Spot vs On-Demand

| Mode | Flag | Savings | Risk |
|---|---|---|---|
| Spot (default) | — | 60-90% | Instance can be reclaimed → training restarts from scratch |
| On-demand | `--no-spot` | 0% | Guaranteed completion, no interruptions |

**Recommendation**: Use `--no-spot` when:
- Training takes >30 min (high restart cost)
- Running during peak hours (EU business hours = high interruption rate)
- You need predictable completion time

Spot is fine for quick experiments (<15 min per job) or off-peak hours.

### Troubleshooting SageMaker Training Jobs

| Symptom | Cause | Fix |
|---------|-------|-----|
| `TypeError: Cannot convert numpy.ndarray to numpy.ndarray` | Old container (py39) has numpy/pandas ABI mismatch when pip upgrades numpy | Use `sagemaker-scikit-learn:1.4-2-py312-cpu-py3` container (already default in skill) |
| `IncompleteRead` / `ProtocolError` during pip install | Transient network error downloading large packages | Resubmit — it's a transient failure |
| `Failed to use ray for memory safe fits` | Ray not installed → sequential fold training (5-10x slower) | Ensure `ray>=2.10.0,<2.45.0` in requirements.txt (already default) |
| Job status stuck at `Training` for >1h with no new logs | Spot instance silently reclaimed or zombie process | Stop job and resubmit with `--no-spot` |
| `ImportError: numpy.core.multiarray` | numpy 2.x installed in py39 container breaks pre-compiled C extensions | Use py312 container (already default) |
| Credentials expired mid-monitoring | `aws-vault` session expired | Re-run `aws-vault exec` in your shell and retry |

### Container Choice Rationale

The skill uses `sagemaker-scikit-learn:1.4-2-py312-cpu-py3` because:
1. **Python 3.12** — compatible with AutoGluon 1.4+ and modern numpy/pandas
2. **scikit-learn 1.4** pre-installed — no ABI conflicts when pip adds autogluon
3. **pip install is fast** — fewer packages need upgrading vs the old py39 container
4. **Ray works out of the box** — enables AutoGluon's parallel fold training (3-5x speedup)

### Time Estimates (ml.m5.2xlarge, presets=good, time-limit=60)

| Phase | Duration | Notes |
|-------|----------|-------|
| Container startup | ~2 min | Instance provisioning + image pull |
| pip install (AutoGluon + ray) | ~3-5 min | ~500 MB of packages |
| Training (multi-label, 6 labels) | ~5-8 min | With ray: ~1 min/label; without: ~4 min/label |
| **Total per job** | **~10-15 min** | On-demand, no interruptions |
| **Total (3 parallel jobs)** | **~10-15 min** | Same — they run simultaneously |

---

## RUL-Specific Experimentation Guide

For **C-MAPSS** and similar run-to-failure datasets, the following techniques have been empirically validated to produce the best results with the `pdm.rul.RULPredictor`:

### Proven Recipe (achieves ~11.4 RMSE on FD001)

1. **Drop constant sensors**: Remove sensors with zero/near-zero variance (FD001: s1, s5, s6, s10, s16, s18, s19). The model does this automatically with `drop_constant_sensors=True`.

2. **Window size = 30, stride = 1**: Use the full window for final training. During HPO, use stride=3 or stride=5 for faster iteration.

3. **Feature selection (top 120)**: Extract all 14 features per sensor (238 total for 17 sensors), then select top 120 by LightGBM feature importance. This reduces overfitting dramatically.

4. **Optuna backend with KFold CV**: Use `backend="optuna"` with 50+ trials. The model uses KFold cross-validation internally to avoid overfitting to a single validation split.

5. **XGBoost + LightGBM in model pool**: Both are competitive; ensemble of diverse models helps.

6. **Multi-seed averaging**: Training with 3-5 random seeds and averaging predictions reduces variance by ~5%.

### Example Training Command

```python
from pdm.rul.model import RULPredictor

model = RULPredictor(window_size=30, rul_cap=125)
result = model.train(
    train_df, test_df,
    backend="optuna",
    n_trials=100,
    stride=1,
    time_limit=600,
    drop_constant_sensors=True,
)
# Expected: RMSE ~11.4 on FD001
```

### Key Lessons Learned

| Hypothesis | Result | Impact |
|-----------|--------|--------|
| Drop constant sensors | Slight improvement alone, critical for feature selection | Enabler |
| Feature selection (238→120) | **Major breakthrough** — reduced RMSE from ~12.8 to ~11.4 | +12% |
| GroupKFold CV during HPO | Prevents overfitting to val split, more reliable selection | +5% |
| LightGBM in model pool | Competitive with XGBoost, enables faster HPO | Diversity |
| Multi-seed averaging | Small but consistent improvement (~0.1-0.3 RMSE) | +2% |
| Exponential smoothing | No improvement on FD001 (noise not the problem) | 0% |
| Cross-sensor interactions | No improvement (window features already capture this) | 0% |
| Stacking meta-learner | Worse than simple weighted ensemble | Negative |

### What Didn't Work

- **Exponential smoothing**: FD001 sensors are already smooth from simulation
- **Cross-sensor interaction features**: Window statistics already capture these patterns
- **Stacking (Ridge meta-learner)**: Overfits to the small ensemble, simple averaging works better
- **Window sizes < 25**: Too little context for degradation trends
- **160+ features**: Diminishing returns, starts overfitting again
