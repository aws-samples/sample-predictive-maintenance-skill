---
name: predictive-maintenance
description: "End-to-end Predictive Maintenance: explores raw data in S3, builds a feature engineering pipeline, trains an AutoML model (RUL regression, failure classification, multi-label classification, or survival analysis), and deploys as a real-time endpoint or daily batch inference job. Use when asked to build a failure prediction model, predict equipment failures, train on IoT sensor data, or estimate remaining useful life."
compatibility: Requires Python 3.11+, uv, boto3, pyarrow, pandas, numpy, scikit-learn, autogluon.tabular
---

# Predictive Maintenance Model Generator

Generate a complete predictive maintenance (PdM) model from raw data stored in S3.

## When This Skill Activates

Use this skill when the user:
- Asks to build or train a predictive maintenance model
- Asks to predict equipment failures or estimate remaining useful life
- Mentions predictive maintenance, RUL, failure classification, multi-label classification, or survival analysis with raw data
- Asks to train a model on IoT sensor data for condition monitoring
- Asks to predict multiple failure modes simultaneously
- Asks to benchmark against known PdM datasets (C-MAPSS, AI4I, NASA Battery, etc.)
- Asks about time-to-failure with censored data or survival curves

## Bundled Assets

```
<SKILL_DIR>/
├── SKILL.md
├── references/
│   ├── data-exploration.md       # Phase 2: bucket exploration, schema, joins, questions
│   ├── benchmark-datasets.md     # Phase 2B: local/benchmark dataset preparation
│   ├── raw-aggregation.md        # Phase 3: S3 loading, joining, label construction, temporal split
│   ├── baselines.md              # Phase 4: folder structure, AD + FP baselines, quality gates
│   ├── experimentation.md        # Phase 5: Hypothesis loop, runtime.py rules, interpretability
│   ├── artifacts.md              # Phase 6: S3 artifact layout contract
│   ├── deployment.md             # Phase 8: endpoint + batch inference + CDK infrastructure
│   └── readme-template.md        # Phase 9: 9-section README structure
├── scripts/
│   ├── setup.sh                     # One-command project setup
│   ├── save_model.sh                # Phase 6: upload artifacts to S3
│   └── pyproject.toml               # Project dependencies
├── sagemaker_container/
│   ├── Dockerfile               # Custom container (Python 3.12 + AutoGluon + lifelines + Flask)
│   ├── serve.py                 # Flask server (/ping + /invocations)
│   ├── inference.py             # Legacy handlers (still works for AutoGluon-only models)
│   └── requirements.txt         # Reference only (deps baked in Dockerfile)
├── infrastructure/
│   ├── app.py                   # CDK entry point
│   ├── batch_inference_stack.py # Stack: EventBridge + Lambda + SageMaker roles
│   ├── lambda/trigger.py        # Lambda handler (creates Processing Job)
│   ├── cdk.json                 # Configurable context variables
│   ├── requirements.txt         # CDK dependencies
│   └── README.md                # Deployment instructions
└── pdm/                         # Standalone PdM library
    ├── __init__.py                  # Public API: all 4 model classes + DatasetMeta
    ├── base.py                      # PDMModel ABC, TrainResult, PredictionResult
    ├── training_time.py             # Phase 5: retraining time estimation
    ├── example_inference.py         # Phase 7: reusable single-day inference + explanations
    ├── data/
    │   ├── __init__.py
    │   ├── utils.py                 # Data loading & feature utilities
    │   ├── data_exploration.py      # S3/parquet utilities (Path A)
    │   ├── dataset_schema.py        # DatasetMeta dataclass
    │   ├── feature_extraction.py    # tsfresh wrapper with manual fallback
    │   └── interactions.py          # Auto-generate pairwise feature interactions
    ├── anomaly_detection/           # AnomalyDetector (Isolation Forest) + TemporalAnomalyDetector (PCA reconstruction) + SpectralResidualDetector (FFT saliency)
    │   ├── __init__.py
    │   ├── model.py                 # AnomalyDetector(PDMModel) — point-wise IF
    │   ├── temporal.py              # TemporalAnomalyDetector(PDMModel) — sliding-window PCA reconstruction
    │   ├── spectral_residual.py     # SpectralResidualDetector(PDMModel) — frequency-domain saliency (BEST for SMAP-like data)
    │   ├── train_anomaly.py         # Legacy CLI (still works)
    │   ├── evaluate_anomaly.py      # AD evaluation + synthetic injection test
    │   ├── synthetic_anomalies.py   # Synthetic anomaly injection utility
    │   └── inference_anomaly.py     # Legacy AD inference CLI
    ├── fault_prediction/            # FailureClassifier (binary + multi-label)
    │   ├── __init__.py
    │   ├── model.py                 # FailureClassifier(PDMModel)
    │   ├── validate_dataset.py      # Dataset validation gate
    │   ├── train.py                 # AutoGluon training CLI (auto-detects formulation)
    │   └── inference.py             # Legacy inference CLI
    ├── rul/                         # RULPredictor (sliding window + AutoGluon)
    │   ├── __init__.py
    │   ├── model.py                 # RULPredictor(PDMModel)
    │   └── train.py                 # CLI
    ├── survival/                    # SurvivalPredictor (Cox PH, Weibull AFT, RSF)
    │   ├── __init__.py
    │   ├── model.py                 # SurvivalPredictor(PDMModel)
    │   └── train.py                 # CLI
    ├── deployment/
    │   ├── __init__.py
    │   ├── batch.py                 # Batch inference utilities
    │   └── serve.py                 # Unified SageMaker handler (all formulations)
    └── remote/                      # SageMaker Training Job submission
        ├── __init__.py
        ├── submit.py                # Single SageMaker training job submission
        ├── train_remote.py          # Container entry script (RUL/classification/survival)
        ├── train_multilabel.py      # Container entry script (multi-label classification)
        └── parallel.py              # Dispatch N experiments, monitor progress, compare
```

## Execution Log (`log.md`)

Immediately create `./log.md` and append **before and after every meaningful action**:

```bash
echo "- [$(date +%H:%M)] EMOJI MESSAGE" >> log.md
```

Emojis: ✅ Completion, ⚠️ Warning, ❌ Error, 🔧 Fix applied, 📊 Decision, 💡 Suggestion, 🧪 Experiment.

**Log these events (non-exhaustive):**
- Action start/completion
- Data loaded (table name, shape)
- Join validated (overlap %, type mismatches)
- Experiment started (name, hypothesis in one line)
- Experiment completed (key metric, delta from previous best)
- Decisions made (keep/discard experiment, re-prioritization)
- Every failure MUST be logged before attempting the fix

**If in doubt, log it.** A verbose log is better than a silent one.

---

## Prerequisites — Gather from the User

Before starting any phase, determine the data path:

**Ask**: "Where is your training data?"

- **Path A (S3 Bucket)**: User provides an S3 bucket URI with raw operational/IoT data
  - Ask for: input data bucket, output model bucket
  - **Treat input bucket as read-only** — never write to it.
  - After Phase 1, proceed to Phase 2A (Data Exploration)

- **Path B (Local/Benchmark Dataset)**: User points to a local folder with data + docs
  - Ask for: folder path, desired formulation (or auto-detect from data)
  - After Phase 1, proceed to Phase 2B (Benchmark Dataset Preparation)

Do NOT proceed until the data path is determined. Record the choice in `log.md`.

---

## Nine-Phase Workflow

Each phase produces specific output files. **Skip any phase whose output files already exist.**

| Phase | Produces | Skip if present |
|-------|----------|-----------------|
| 1. Setup | `pyproject.toml`, scripts | `pyproject.toml` exists |
| 2A. Data Exploration & Strategy | `./data_exploration.md` | File exists AND contains `## User Decisions` |
| 2B. Benchmark Dataset (Path B) | `./data/dataset_meta.json` | File exists with `"source": "benchmark"` |
| 3. Raw Data Aggregation | `./pdm/raw_dataset.py`, `./data/raw_train.csv`, `./data/raw_test.csv` | `./data/raw_train.csv` exists |
| 4. Baselines | `./*/baseline/model/metrics.json` | metrics.json exists per model |
| 5. Experimentation | `./<model_type>/experiments.md` updated | experiments complete |
| 6. Post-Training Plan & Save | `./post_training_plan.md`, artifacts on S3 | User declines or confirms |
| 7. Example Inference | `./data/example_predictions.csv` | User declines or file exists |
| 8. Deploy to SageMaker | Endpoint and/or batch job | User declines or endpoint already InService |
| 9. Documentation | `./README.md` | File exists |

**Path A uses Phases 2→3→4+. Path B uses Phase 2B→4+ (skips 2 and 3).**

**Before starting, check which outputs exist and begin from the earliest incomplete phase.**

**CRITICAL: Phase 2 ends with questions for the user. Do NOT proceed to Phase 3 until the user answers.** If `data_exploration.md` exists but has empty `[Answer]:` tags or no `## User Decisions` section, present the questions and wait.

---

## Phase 1: Setup

```bash
bash <SKILL_DIR>/scripts/setup.sh
```

Where `<SKILL_DIR>` is the directory containing this SKILL.md file.

Creates venv, installs dependencies, copies scripts + libraries. Then create `log.md`.

⚠️ If `uv run` fails after setup, use `.venv/bin/python` directly — the venv is fully functional after `setup.sh` completes.

---

## Phase 2A: Data Exploration & Strategy

**Read [references/data-exploration.md](references/data-exploration.md) for full guidelines** — covers bucket exploration, schema analysis, join validation, question format, and decision recording.

Steps: discover bucket structure → explore schemas → validate join keys → write `data_exploration.md` → present questions → **STOP AND WAIT** → validate answers → record decisions.

**Formulation recommendation**: When asking the user which formulation to use, explain the tradeoffs in plain language (see `pdm/README.md` § "Choosing a Formulation"). Use the decision flowchart:
- Run-to-failure data + need precise timeline → RUL
- Run-to-failure data + need alerts only → Classification
- Devices maintained before failure (censored data) → Survival
- Multiple independent failure modes → Multi-label classification

### MANDATORY: Stop and Ask Before Phase 3

**CRITICAL: Phase 2 MUST end by presenting questions to the user and STOPPING. Do NOT proceed to Phase 3 until the user explicitly answers.** Even if the user's initial prompt specifies labels or target tables, the following decisions require explicit user confirmation because they have significant impact on model design and cannot be reliably inferred:

**Always ask (even when the user provides some context in their initial prompt):**

1. **Formulation** — confirm the recommended formulation (multi-label, binary, RUL, survival). Propose based on data evidence but get explicit confirmation.
2. **Prediction window (horizon)** — how many days ahead should the model predict? This is an operational decision only the user can make (e.g., 7 days, 14 days, 30 days). It depends on maintenance lead times, not data properties.
3. **Feature scope** — which tables/data sources to use as input features? List all available tables and ask the user to confirm which ones to include. Do NOT assume — some tables may be supplementary context only, not model inputs.
4. **Optimization priority** — does the user prioritize recall (catch all failures), precision (minimize false alarms), or balanced F1? This affects training eval metric and threshold selection.
5. **Anomaly detection** — also build an unsupervised model? Recommend based on data dimensionality.

**The agent MUST present these questions and STOP — do not fill in answers by assumption, even if the user's initial prompt seems to imply them.** The user may have additional context (operational constraints, business requirements, lead times) that changes the "obvious" answer.

If `data_exploration.md` exists but has empty `[Answer]:` tags or no `## User Decisions` section, present the questions and wait.

---

## Phase 2B: Benchmark Dataset Preparation (Path B only)

**Read [references/benchmark-datasets.md](references/benchmark-datasets.md) for full guidelines** — covers auto-detection, adapter writing, validation, column conventions.

Use when the user provides a local folder with data + documentation instead of an S3 bucket.

Steps: try auto-detection → confirm formulation with user → write adapter or use built-in loader → validate output → proceed to Phase 4.

**Output**: `./data/raw_train.csv`, `./data/raw_test.csv`, `./data/dataset_meta.json`

**Skips Phases 3 entirely.**

---

## Phase 3: Raw Data Aggregation

**Read [references/raw-aggregation.md](references/raw-aggregation.md) for full guidelines** — covers gotchas, key functions, data loading patterns, joining, label construction, and temporal split.

Read `data_exploration.md` (including User Decisions), write `pdm/raw_dataset.py`, then run:

```bash
uv run python pdm/raw_dataset.py
```

Produces `./data/raw_train.csv` and `./data/raw_test.csv` — the **single source of truth** for all experiments.

Log raw dataset shape, column count, label positive rates, and split sizes in `log.md`.

---

## Phase 4: Baselines

**Read [references/baselines.md](references/baselines.md) for full guidelines** — covers folder structure, anomaly detection baseline (Isolation Forest), fault prediction baseline (runtime.py template, training commands), and quality gates.

Build a baseline for the model(s) appropriate to the formulation. Read `dataset_meta.json` (if present) or detect formulation from the data columns:

| Formulation | Model Family | Train Command |
|---|---|---|
| `rul` | RUL Predictor | `uv run python pdm/rul/train.py --train ./data/raw_train.csv --test ./data/raw_test.csv` |
| `classification` / `multilabel` | Failure Classifier | `uv run python pdm/fault_prediction/train.py --train ... --test ...` |
| `survival` | Survival Predictor | `uv run python pdm/survival/train.py --train ... --test ...` |
| (if requested) | Anomaly Detection | `uv run python pdm/anomaly_detection/train_anomaly.py --train ... --test ...` |

**Anomaly Detection is built only if the user opts in** — asked in Phase 2A (Path A) or derived from the data folder documentation (Path B).

Log baseline metrics in `log.md`.

---

## Phase 5: Experimentation

**Read [references/experimentation.md](references/experimentation.md) for full guidelines** — covers experiment structure, hypothesis formation, CAAFE-style reasoning, and stopping criteria.

Applies to **all active models** (primary formulation + anomaly detection if requested). Each model gets its own `experiments.md` and `experiments/` folder following the same convention.

### Overview

For each active model (e.g., `anomaly_detection/`, `fault_prediction/`):

1. **Build Knowledge Base** — Use the **knowledge-acquisition** skill (`.kiro/skills/knowledge-acquisition/SKILL.md`) to research the specific formulation and build a `./wiki/` with state-of-the-art techniques and domain insights. Experiments are driven by this knowledge.
2. **Propose** — Write `./<model_type>/experiments.md` with 3–5 ranked hypotheses drawn from wiki research
3. **User Review** — Present plan, wait for approve / edit / skip
4. **Loop** (for each experiment 01+):
   - A) Hypothesize — grounded in wiki knowledge + baseline analysis
   - B) Implement — write `runtime.py` with feature preparation logic
   - C) Train & Verify — train model, verify inference
   - D) Log — write README, update experiments.md
5. **Combine & Build Final** — identify all experiments that beat baseline, merge their improvements into a combined `runtime.py`, train the combined model, verify it meets or exceeds the best individual, ask user about final retraining (extended rebuild recommended; use `pdm.training_time.format_retraining_options()` for time estimates)

### MANDATORY: Stop and Ask Before Running Experiments

**CRITICAL: After writing the experiment plan in `experiments.md`, you MUST present it to the user and STOP. Do NOT run any experiment until the user explicitly approves, edits, or skips.** The user may want to reprioritize, add domain-specific hypotheses, or skip experimentation entirely.

Present the plan with these options:
- A) **Approve** — run experiments in the proposed order
- B) **Edit** — user will modify experiments.md
- C) **Skip** — use the baseline as final model

### Experiment Examples by Model Type

**Anomaly Detection** experiments might test:
- Different contamination values or threshold strategies
- Feature subsets (e.g., only electrical sensors vs. all)
- Alternative algorithms (LOF, Autoencoder)
- Different normal-only filtering criteria

**Fault Prediction** experiments might test:
- Domain-informed interaction features (CAAFE-style)
- Rolling window representations
- Feature selection based on importance analysis
- Different temporal aggregation windows

### Quality Gates

| Model | Criterion | Threshold |
|-------|-----------|-----------|
| Anomaly Detection (labels) | AUROC | ≥ 0.75 |
| Anomaly Detection (no labels) | Synthetic detection rate | ≥ 80% |
| Classification | Test F1 | ≥ 0.50 |
| Multi-label | Median test F1 | ≥ 0.50 |
| RUL | Test RMSE | < 50% of RUL range |
| RUL | NASA score (normalized) | < 1.0 |
| Survival | Concordance index | > 0.65 |
| Survival | Integrated Brier score | < 0.25 |

### Stopping Criteria

Stop when quality gate passes AND last experiment improved <2%, OR max 5 experiments reached.

---

## Phase 6: Post-Training Plan & Save to S3

After experimentation is complete and the combined model is built, gather the post-training plan from the user.

### MANDATORY: Stop and Ask Before Saving or Deploying

**CRITICAL: Do NOT save to S3, run inference, or deploy without explicit user confirmation.** Write `post_training_plan.md` with the questions below, tell the user, and STOP. The user decides what happens next.

1. Write a `./post_training_plan.md` file with the questions below (substituting actual values for placeholders).
2. Tell the user: *"I've written `post_training_plan.md` with 3 questions about next steps. Please fill in the `[Answer]:` fields in that file and let me know when you're done."*
3. **STOP AND WAIT.** Do NOT summarize the questions in chat or ask them conversationally — the file IS the interface.
4. When the user says they've answered, re-read `post_training_plan.md` and proceed based on their answers.

```markdown
# Post-Training Plan

## Question 1: Save model to S3
**Do you want to save this model to S3?**

A) **Yes** — save all project artifacts to S3
B) **No** — keep locally only

If yes, which S3 bucket? (Leave blank to use the input data bucket)

[Bucket]:
[Answer]:

## Question 2: Example inference
**Do you want to run inference on a day of device telemetry to generate an example output for evaluation?**

This produces a predictions CSV showing what the model would output in production, so you can evaluate quality before deploying.

A) **Yes, with explanations** — include per-prediction feature contributions (top features driving each prediction)
B) **Yes, predictions only** — faster, no explainability
C) **No** — skip example inference

Proposed inference date: [LAST_TELEMETRY_DATE] (the last day with telemetry data)

[Date]: (leave blank to accept proposed date, or enter YYYY-MM-DD)
[Output bucket]: (where to store inference results; leave blank to use the model bucket above)
[Answer]:

## Question 3: Deployment
**Do you want to deploy this model?**

A) **Real-time endpoint** — SageMaker endpoint for prediction-by-prediction API calls
B) **Batch inference** — daily scheduled job processing yesterday's telemetry (EventBridge + Lambda + SageMaker Processing Job)
C) **Both** — endpoint + daily batch
D) **No** — do not deploy

[Answer]:
```

Replace `[LAST_TELEMETRY_DATE]` with the actual last date from the telemetry partition range discovered in Phase 2.

After the user answers:
1. Record decisions in `post_training_plan.md`
2. If Save = Yes: run `bash save_model.sh <bucket-name>` to upload artifacts to `s3://<bucket>/YYYYMMDD_HHMM/`
3. Proceed to Phase 7 if Example Inference = Yes, otherwise skip to Phase 8

### What `save_model.sh` Uploads

The script uploads artifacts organized by concern — see `references/artifacts.md` for the full contract:

```
s3://<bucket>/<YYYYMMDD_HHMM>/
├── dataset/          raw_dataset.py + raw CSVs (~50-200 MB)
├── training/         runtime.py + train_config.json + pdm/ library (~600 KB)
├── model/            ag_model/ + metadata.json + metrics.json (~1-1.5 GB)
├── inference/        inference.py + Dockerfile + serve.py (~10 KB)
├── infrastructure/   CDK stack + Lambda trigger (~20 KB)
└── README.md
```

**NOT uploaded**: processed experiment CSVs, discarded experiments, baseline model, caches, venv, `.git/`, out-of-fold predictions (`oof.pkl`).

The script auto-detects the winning model by finding the newest `metrics.json` under `fault_prediction/`. Override with: `bash save_model.sh <bucket> <model-dir>`.

**Why AutoGluon models are ~1 GB for multi-label**: AutoGluon stores a copy of training features (`X.pkl`) inside each label's directory for `feature_importance()`. With 9 labels this is 9 × ~67MB = ~600MB. The fold model weights add ~725MB. Together that's ~1.3GB.

---

## Phase 7: Example Inference

**Skip this phase** if the user answered "No" to example inference in the post-training plan.

Run the combined model on a single day of telemetry to produce a predictions file the user can evaluate.

### Using the reusable script

```bash
.venv/bin/python pdm/example_inference.py \
    --bucket SOURCE_BUCKET \
    --telemetry-prefix telemetry/ \
    --date YYYY-MM-DD \
    --model-dir ./fault_prediction/baseline/model \
    --runtime ./fault_prediction/baseline/runtime.py \
    --output ./data/example_predictions.csv \
    --metadata-prefix device_metadata/ \
    --entity-col device_id \
    --explain
```

Substitute the actual values from the project (bucket, prefixes, entity column, combined model path).

**Key references in `model/metadata.json`:**
- `metadata["feature_names"]` — ordered list of features the model expects
- `metadata["label_names"]` — list of label column names (e.g., `["label_Hardware_fault", ...]`)
- `metadata["formulation"]` — "multilabel", "classification", "rul", or "survival"

**Column alignment:** The script uses `align_to_model()` from `pdm.data.utils` to handle missing/extra columns between inference data and training data. Missing features are filled with 0; extra columns are dropped.

**Explanations:** Uses `feature_contributions()` from `pdm.data.utils` — perturbation-based (zeroing each feature and measuring probability change). Explains only positive predictions using the top-10 features from `metrics.json` feature importance.

### Output format

CSV with columns:

| Column | Description |
|--------|-------------|
| `device_id` | Device identifier |
| `observation_date` | The inference date |
| `label_*_probability` | Predicted probability per label (classification/multi-label) |
| `label_*_prediction` | Binary prediction (1/0) per label |
| `top_features` | (If explanations enabled) Semicolon-separated `feature_name(Δ=±N)` for top features per positive label |

For anomaly detection models, include `anomaly_score`, `is_anomaly`, and `top_anomalous_features`.

### After saving

1. Upload predictions to the same S3 path used in Phase 6: `aws s3 cp ./data/example_predictions.csv s3://<bucket>/<timestamp>/data/example_predictions.csv`
2. Print summary stats (# devices scored, # predicted positive per label)
3. Log to `log.md`
4. Inform the user where the output is stored and proceed to Phase 8

---

## Phase 8: Deploy to SageMaker

**Skip this phase** if the user answered "No" to deployment in the post-training plan.

**Read [references/deployment.md](references/deployment.md) for full guidelines** — covers endpoint deployment, batch inference, and CDK infrastructure.

Execute the deployment mode(s) chosen by the user. Do NOT deploy without the explicit confirmation already obtained in Phase 6.

**Phase 8A** — Custom container → ECR → SageMaker endpoint (Steps: verify versions, build/push, package model.tar.gz, test locally, deploy, verify).

**Phase 8B** — `batch_inference.py` using `pdm.deployment.batch` → SageMaker Processing Job triggered daily by EventBridge + Lambda (CDK in `infrastructure/`). Key: use `predict_proba()` ONLY for speed, `ml.m5.2xlarge` for multi-label models. For anomaly detection, generate `batch_inference_anomaly.py` using Isolation Forest scoring with z-score feature explanations (see deployment reference § 8B.3).

---

## Phase 9: Documentation

**Read [references/readme-template.md](references/readme-template.md)** — 9 sections covering overview, label definition, features, pipeline, training, evaluation, inference, project structure.

Write `./README.md` for both domain experts and developers. Gather information from all previous phases — read the actual code, metrics, deployment artifacts, and inference outputs produced in Phases 1–8. Include:
- A section documenting the experimentation results (all experiments with their hypotheses and metrics)
- Plain-language metric explanations (adapt from `pdm/README.md` § "Understanding the Metrics")
- Explainability: how to interpret SHAP plots and per-prediction feature contributions
- Inference instructions based on the actual example inference output from Phase 7 (if executed)
- Deployment details: endpoint name, batch schedule, how to invoke — from Phase 8 artifacts (if deployed)

---

## Final Project Structure

```
./
├── README.md                     # Phase 9
├── log.md                        # Execution log
├── pyproject.toml
├── data_exploration.md           # Phase 2 (includes User Decisions)
├── post_training_plan.md         # Phase 6: Post-training decisions (save, inference, deploy)
├── wiki/                         # Phase 5: Domain knowledge base (LLM Wiki)
│   ├── index.md
│   ├── concepts/
│   │   ├── feature-engineering-pdm.md
│   │   ├── <formulation>-techniques.md
│   │   └── domain-features.md
│   └── comparisons/
│       └── model-architectures.md
├── deploy_endpoint.py            # Phase 8A: SageMaker endpoint deployment
├── batch_inference.py            # Phase 8B: Daily batch predictions (fault prediction)
├── batch_inference_anomaly.py    # Phase 8B: Daily anomaly detection batch inference
├── infrastructure/               # Phase 8B: CDK app for EventBridge + Lambda + Processing Job
│   ├── app.py
│   ├── batch_inference_stack.py
│   ├── lambda/trigger.py
│   ├── cdk.json
│   └── requirements.txt
├── pdm/
│   ├── __init__.py
│   ├── data/                     # Data loading & exploration submodule
│   │   ├── __init__.py
│   │   ├── utils.py              # Data loading & feature utilities
│   │   └── data_exploration.py   # S3/parquet utilities
│   ├── deployment/               # Batch & real-time inference utilities
│   │   ├── __init__.py
│   │   └── batch.py             # load_telemetry_window, aggregate_telemetry, predict_proba
│   ├── anomaly_detection/        # Unsupervised AD submodule
│   │   ├── __init__.py
│   │   ├── train_anomaly.py      # Isolation Forest AD training
│   │   ├── evaluate_anomaly.py   # AD evaluation + synthetic injection test
│   │   ├── synthetic_anomalies.py # Synthetic anomaly injection utility
│   │   └── inference_anomaly.py  # AD inference CLI
│   ├── fault_prediction/         # Supervised fault prediction submodule
│   │   ├── __init__.py
│   │   ├── validate_dataset.py   # Validation gate
│   │   ├── train.py              # AutoGluon training (auto-detects formulation)
│   │   └── inference.py          # Inference CLI
│   └── raw_dataset.py            # Raw data aggregation (Phase 3)
├── data/
│   ├── raw_train.csv             # Phase 3 output (raw training split)
│   ├── raw_test.csv              # Phase 3 output (raw test split)
│   ├── example_predictions.csv   # Phase 7 output (example inference results)
│   └── _cache_*.parquet          # Cached intermediates
├── anomaly_detection/
│   ├── experiments.md            # Experiment backlog & results
│   ├── baseline/
│   │   ├── runtime.py            # Feature prep + training
│   │   ├── inference.py          # Local inference (last day of test data)
│   │   ├── data/
│   │   │   ├── train.csv
│   │   │   └── test.csv
│   │   ├── predictions.csv       # Latest batch inference output (from S3)
│   │   └── model/
│   │       ├── isolation_forest.joblib
│   │       ├── scaler.joblib
│   │       ├── threshold.json
│   │       ├── metadata.json
│   │       ├── baseline_stats.json
│   │       ├── anomaly_scores_test.csv
│   │       ├── score_distribution.png
│   │       └── metrics.json
│   └── experiments/
│       └── 01_<name>/
│           └── model/
└── fault_prediction/
    ├── experiments.md            # Experiment backlog & results
    ├── baseline/
    │   ├── README.md
    │   ├── runtime.py
    │   ├── data/
    │   │   ├── train.csv
    │   │   └── test.csv
    │   └── model/
    │       ├── ag_model/
    │       ├── metadata.json
    │       ├── metrics.json
    │       └── shap_summary.png
    └── experiments/
        ├── 01_<name>/
        │   ├── README.md
        │   ├── runtime.py
        │   ├── data/
        │   └── model/
        ├── ...
        └── combined/           # Merged improvements from all winning experiments
            ├── README.md
            ├── runtime.py
            ├── data/
            └── model/
```
