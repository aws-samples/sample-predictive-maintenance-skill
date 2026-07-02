# README Template

Write a `./README.md` that serves two audiences:
1. **Domain experts** who need to validate the approach (data choices, label logic, feature rationale)
2. **Developers** who need to reproduce, retrain, and operate the model

## Required Sections

The README must cover these sections (use the actual values from the project):

### 1. Project Overview
- What the model predicts (formulation, target variable)
- Data source (S3 bucket, tables used)
- Model performance (test metric from `model/metrics.json`)

### 2. Label Definition (for domain expert review)

This section must allow a domain expert to answer: *"Is this the right definition of failure?"*

- **Source table** and column(s) used to derive the label
- **Exact logic**: what rows count as positive (failure), what counts as negative (healthy)
- **Positive examples**: list 3–5 concrete issue types/codes that map to label=1, with their descriptions
- **Negative definition**: explicitly state what "healthy" means (no records in health table? no open cases? etc.)
- **Population**: how many devices total, how many positive, what's the positive rate
- **Known limitations**: what failure modes might be MISSING from this label (e.g. devices that failed but weren't flagged in the health system, devices with no telemetry)
- **Temporal definition**: is this "has ever failed" or "failed within N days of the telemetry window"?

### 3. Input Data & Feature Rationale (for domain expert review)

This section must allow a domain expert to answer: *"Are we feeding the model the right signals?"*

- **Data scope**: what time window of data is used, how many devices, what partitions
- **Feature sources**: for each source table, explain WHY it was chosen and what physical/business signal it represents
- **Sensor selection rationale**: why these specific sensors were chosen (link to known failure modes where possible)
- **Aggregation strategy**: why mean/std/max/min? What time granularity? Per device or per device-day?
- **Device metadata rationale**: why device age, hardware revision, etc. matter for failure prediction
- **What's excluded and why**: tables or columns that exist in the bucket but were deliberately NOT used (e.g. case_emails excluded because NLP features not in scope)
- **Coverage**: what % of the target population has telemetry? What % has device metadata? What happens to devices with missing features?

### 4. Data Processing Pipeline

- How `pdm/dataset.py` works (data loading → feature engineering → label construction → merge → validate → save)
- **Join strategy**: which tables are joined on which keys, in what order, with what join type (inner/left)
- **Join validation results**: overlap percentages for each join (from Phase 2 exploration)
- **Type coercions**: any key casting (e.g. int64→string) and why
- **Deduplication logic**: how duplicates in source tables are handled
- **Filtering applied**: what rows/sensors/devices are filtered out and why
- Location of training data: `./data/dataset.csv`
- Location of cached intermediates: `./data/_cache_*.parquet`
- How to regenerate: `uv run python pdm/dataset.py`

### 5. Feature Engineering (`pdm/runtime.py`)
- List of all features the model expects (group by source)
- For each feature source: what raw data is needed, where it comes from, and how it's transformed
- How to gather runtime features for a **new prediction** (what data to query + how to call `runtime.py`)
- Example code snippet showing how to produce a feature vector for inference

### 6. Training
- How to re-train: `uv run python train.py --data ./data/dataset.csv --output ./model`
- What `train.py` does (AutoGluon TabularPredictor, model selection, metrics output)
- How to change hyperparameter budget (`--n-trials`, `--timeout`)

### 7. Evaluation
- Where metrics are stored: `./model/metrics.json`
- How to interpret them (F1/precision/recall for classification, RMSE for RUL, concordance for survival)
- Feature importance (top 10 features from `metrics.json`) with brief interpretation of what each signal means physically
- Precision/recall tradeoff: what the current operating point implies (e.g. "87% of actual failures are caught, but 33% of alerts are false positives")

### 8. Inference
- **Read the actual inference module** (e.g. `pdm/fault_prediction/inference.py`) and use the real exported class/function names in the example — do NOT invent function names
- How to load the trained model and run predictions
- What the output means (0/1 for classification, probability for ranking)
- Example code using real imports that exist in the codebase

### 9. Project Structure
- Full directory tree with one-line descriptions

## Guidelines

- The domain expert sections (2, 3) should be readable WITHOUT understanding code — use plain language, concrete examples, and domain terminology
- Use concrete values (actual feature names, actual S3 paths, actual metric scores) — not placeholders
- Reference `pdm/runtime.py` functions by name so developers know exactly what to call
- If a decision was made that a domain expert might disagree with, call it out explicitly (e.g. "We used a binary label rather than per-issue-type because...")
