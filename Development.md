# Development Guide

## Project Structure

```
predictive-maintenance/
├── SKILL.md                  # Agent skill definition (8-phase pipeline)
├── pyproject.toml            # Dependencies + pytest configuration
├── pdm/                      # Core library (the code we ship)
│   ├── __init__.py           # Public API: PDMModel, AnomalyDetector, FailureClassifier, etc.
│   ├── base.py               # PDMModel ABC, TrainResult, PredictionResult
│   ├── data/                 # Data loading, exploration, benchmark loaders
│   ├── anomaly_detection/    # Isolation Forest wrapper
│   ├── fault_prediction/     # AutoGluon classification (binary + multi-label)
│   ├── rul/                  # Sliding window features + AutoGluon regression
│   ├── survival/             # Cox PH, Weibull AFT, Random Survival Forest
│   ├── deployment/           # Batch inference + SageMaker serving
│   └── remote/               # SageMaker Training Job submission
├── tests/                    # Unit & integration tests (< 1 min)
│   ├── conftest.py           # Fixtures loading from tests/data/
│   ├── data/                 # Small mock CSVs (committed, ~20 KB total)
│   └── test_*.py             # Test modules
├── benchmarks/               # Performance benchmarks (separate from tests)
│   ├── baselines.json        # Locked baseline metrics
│   └── run_all.py            # Benchmark runner script
├── infrastructure/           # CDK stack for batch inference
├── sagemaker_container/      # Custom Docker container
├── scripts/                  # Setup and model upload scripts
└── references/               # Phase-specific documentation for the agent
```

---

## Testing

Tests validate **our library code** — not third-party libraries (AutoGluon, lifelines, scikit-survival). We test:
- Data transformations and feature engineering (sliding window, EAV aggregation)
- Interface contracts (train → returns TrainResult, predict → returns PredictionResult)
- Serialization roundtrips (save → load → predict produces same results)
- Formulation auto-detection logic
- Scoring functions (NASA asymmetric scoring)
- Edge cases and error handling


### Mock Data

All test data lives in `tests/data/` as small committed CSV files:

| File | Rows | Purpose |
|------|------|---------|
| `rul.csv` | 150 | 5 units × 30 cycles, 4 sensors, RUL target |
| `classification.csv` | 100 | 5 features, 10% failure rate |
| `survival.csv` | 30 | 6 features, duration + event |
| `anomaly.csv` | 100 | 6 sensors, 10% injected spikes |

These datasets are tiny enough that even model training completes in seconds.

### Running Tests

```bash
# Full suite (should complete in < 90 seconds)
uv run pytest tests/

# Specific module
uv run pytest tests/test_rul.py

# Verbose with short tracebacks
uv run pytest tests/ -v --tb=short

# Stop on first failure
uv run pytest tests/ -x
```

### Test Modules

| Module | What It Tests |
|--------|---------------|
| `test_base.py` | PredictionResult, TrainResult, formulation detection, model registry |
| `test_rul.py` | `sliding_window_features()`, `nasa_scoring()` |
| `test_anomaly_detection.py` | AnomalyDetector train/predict/explain/save/load |
| `test_fault_prediction.py` | FailureClassifier formulation detection, feature exclusion |
| `test_survival.py` | SurvivalPredictor train/predict/save/load |
| `test_data_loading.py` | DatasetMeta, benchmark loader detection, load_or_cache |
| `test_deployment.py` | Prediction JSON serialization, response format |

### Adding Tests for New Features

When implementing a new feature:

1. Write tests FIRST for the new module (TDD)
2. Run existing tests to confirm no regression: `uv run pytest tests/`
3. Implement the feature
4. Run tests again — all must pass
5. If a test needs updating (e.g., new field in PredictionResult), update it explicitly and document why

---

## Benchmarking

Benchmarks are **separate from tests**. They:
- Use real datasets (pre-downloaded, not committed to git)
- Train models with real time budgets (2+ minutes each)
- Produce quantitative metrics compared against locked baselines
- Run on-demand before merging (not on every commit)

### Locked Baselines

`pdm/benchmarks/baselines.json` contains the metrics each benchmark must meet:

| Benchmark | Dataset | Metric | Baseline | Tolerance |
|-----------|---------|--------|----------|-----------|
| `cmapss_fd001_rul` | C-MAPSS FD001 | RMSE ↓ | 16.48 | ±10% |
| `ai4i_classification` | AI4I 2020 | F1 ↑ | 0.82 | ±5% |
| `battery_survival` | NASA Battery | C-index ↑ | 1.00 | ±5% |
| `smap_anomaly_detection` | NASA SMAP | F1 ↑ | 0.54 | ±10% |

Tolerances account for run-to-run variance from random seeds and model selection.

### Running Benchmarks

```bash
# Run all benchmarks (data auto-downloads if missing, ~5 minutes total)
uv run python -m pdm.benchmarks.benchmark ./benchmark_data all

# Run a single benchmark
uv run python -m pdm.benchmarks.benchmark ./benchmark_data cmapss
uv run python -m pdm.benchmarks.benchmark ./benchmark_data ai4i
uv run python -m pdm.benchmarks.benchmark ./benchmark_data battery
uv run python -m pdm.benchmarks.benchmark ./benchmark_data smap

# Update baselines after confirmed improvement
uv run python -m pdm.benchmarks.benchmark ./benchmark_data all --update

# Extended: 3 seeds, report mean ± std
uv run python -m pdm.benchmarks.benchmark ./benchmark_data all --extended

# Download data only (without running benchmarks)
uv run python -m pdm.benchmarks.download ./benchmark_data all
```

### Benchmark Data Setup

Benchmark data is auto-downloaded on first run. You can also download manually:

```bash
uv run python -m pdm.benchmarks.download ./benchmark_data all
```

Data sources:
- **C-MAPSS**: [NASA Data Portal](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
- **AI4I 2020**: [UCI ML Repository](https://archive.ics.uci.edu/dataset/601)
- **NASA Battery**: [NASA PCoE Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) (Li-Ion charge/discharge cycles)
- **NASA SMAP**: Spacecraft telemetry anomaly detection

### When to Run Benchmarks

| Trigger | Run Tests? | Run Benchmarks? |
|---------|-----------|----------------|
| Every save / commit | ✅ | ❌ |
| Before merge / PR | ✅ | ✅ |
| After implementing a new feature | ✅ | ✅ (verify improvement or no regression) |
| Weekly check | ✅ | ✅ `--extended` |

---

## Feature Development Workflow

When adding a new capability to the library:

```
1. Write new tests:     tests/test_{new_module}.py
2. Run existing tests:  uv run pytest tests/       → all pass (no regression)
3. Implement feature:   pdm/{new_module}/
4. Run all tests:       uv run pytest tests/       → all pass
5. Run benchmarks:      uv run python benchmarks/run_all.py
6. If improvement:      uv run python benchmarks/run_all.py --update
7. Commit
```

### Acceptance Criteria

Every new feature must satisfy:

1. **No regression**: All existing tests pass. All benchmarks remain within tolerance.
2. **Quantitative improvement** (if the feature modifies a model pipeline): At least one benchmark metric improves beyond the tolerance band.
3. **Net new features** (additive modules that don't change existing pipelines): Only need to prove no regression — no improvement target required.

Examples of "must improve" features: new feature engineering, new model architectures, better algorithm selection.
Examples of "net new" features: drift detection, new benchmark loaders, uncertainty intervals, foundation model embeddings.

---

## Installation

### Development Setup

```bash
# Clone the skill
git clone <repo-url> ~/.kiro/skills/predictive-maintenance
cd ~/.kiro/skills/predictive-maintenance

# Install core + test dependencies
uv sync

# Verify installation
uv run pytest tests/
```

### Optional Extras

Install additional capabilities as needed:

```bash
# Automated time series feature extraction
uv pip install tsfresh>=0.20

# Multiple anomaly detection algorithms
uv pip install pyod>=2.0

# Drift detection for production monitoring
uv pip install alibi-detect>=0.12

# Deep learning models (InceptionTime, PatchTST, ROCKET)
uv pip install tsai>=0.3.9 torch>=2.0

# Foundation model embeddings (zero-shot anomaly detection)
uv pip install chronos-forecasting>=1.4

# Domain adaptation for cross-condition transfer
uv pip install rul-adapt>=0.4 pytorch-lightning>=2.0
```

Or install multiple extras at once:

```bash
uv pip install -e ".[tsfresh,anomaly,monitoring]"
```

### Requirements

- **Python 3.11+**
- **uv** (recommended) or pip
- **AWS credentials** with S3 read access (for data loading and deployment)

---

## Code Style

- Python 3.11+ (type hints, `X | Y` unions)
- All model classes inherit from `PDMModel` ABC
- All training returns `TrainResult`; all prediction returns `PredictionResult`
- New modules must be importable without optional dependencies (use conditional imports)
- Feature columns are auto-selected by excluding known target/metadata columns
