"""Training time estimation for AutoGluon PdM models.

Estimates wall-clock training time based on model configuration, dataset size,
and execution environment. Accounts for multi-label multiplier (one model per label).

Usage:
    from pdm.training_time import estimate_training_minutes

    minutes = estimate_training_minutes(
        formulation="multilabel",
        n_labels=9,
        time_limit=300,
        presets="best_quality",
        n_train_rows=20000,
        n_features=460,
        execution="local",
    )
"""


def estimate_training_minutes(
    formulation: str,
    time_limit: int,
    presets: str = "good",
    n_labels: int = 1,
    n_train_rows: int = 10000,
    n_features: int = 100,
    execution: str = "local",
    instance_type: str = "ml.m5.2xlarge",
    skip_importance: bool = False,
) -> dict:
    """Estimate training wall-clock time in minutes.

    Args:
        formulation: One of "multilabel", "classification", "rul", "survival".
        time_limit: AutoGluon --time-limit in seconds (per label for multilabel).
        presets: AutoGluon presets ("good", "best_quality", "medium_quality").
        n_labels: Number of labels (only relevant for multilabel).
        n_train_rows: Number of training rows.
        n_features: Number of features.
        execution: "local" or "sagemaker".
        instance_type: SageMaker instance type (ignored for local).
        skip_importance: If True, assume --skip-importance is used (no FI cost).

    Returns:
        Dict with keys: total_minutes, breakdown (dict of phase -> minutes),
        description (human-readable summary).
    """
    breakdown = {}

    # --- Per-label training time ---
    # AutoGluon's time_limit is a budget: it may finish early on small data
    # or use the full budget on large data. Estimate actual usage.
    data_factor = _data_complexity_factor(n_train_rows, n_features)
    preset_factor = _preset_factor(presets)

    # Effective time per label: min(time_limit, what data actually needs)
    # For "good" presets, AG often finishes in ~60-70% of budget on medium data
    # For "best_quality", it uses the full budget (ensembling, bagging, stacking)
    effective_seconds_per_label = time_limit * data_factor * preset_factor

    # --- Multi-label multiplier ---
    if formulation == "multilabel":
        # AutoGluon trains labels sequentially (each is an independent predictor)
        training_seconds = effective_seconds_per_label * n_labels
        breakdown["per_label_seconds"] = round(effective_seconds_per_label)
        breakdown["n_labels"] = n_labels
    else:
        training_seconds = effective_seconds_per_label

    training_minutes = training_seconds / 60
    breakdown["model_training"] = round(training_minutes, 1)

    # --- Feature importance (permutation-based) ---
    # train.py calls predictor.feature_importance() per label after training.
    # With --skip-importance, this is skipped entirely.
    if skip_importance:
        fi_minutes = 0.0
    else:
        fi_minutes = _feature_importance_minutes(
            n_features, n_labels, n_train_rows, presets, formulation
        )
    breakdown["feature_importance"] = round(fi_minutes, 1)

    # --- Overhead: data loading, preprocessing, metric computation ---
    overhead_minutes = _overhead_minutes(n_train_rows, n_features, formulation)
    breakdown["overhead"] = round(overhead_minutes, 1)

    # --- SageMaker-specific overhead ---
    if execution == "sagemaker":
        setup = _sagemaker_setup_minutes(instance_type)
        breakdown["container_setup"] = setup["provisioning"]
        breakdown["pip_install"] = setup["pip_install"]
        sagemaker_overhead = setup["provisioning"] + setup["pip_install"]
    else:
        sagemaker_overhead = 0

    total = training_minutes + fi_minutes + overhead_minutes + sagemaker_overhead
    breakdown["total_minutes"] = round(total, 1)

    description = _format_description(
        formulation, n_labels, time_limit, presets, execution, breakdown
    )

    return {
        "total_minutes": round(total, 1),
        "breakdown": breakdown,
        "description": description,
    }


def _data_complexity_factor(n_rows: int, n_features: int) -> float:
    """How much of the time_limit budget AutoGluon actually uses.

    Small datasets finish early; large datasets use the full budget.
    Returns a factor between 0.5 and 1.0.
    """
    # Empirical: AG uses ~50% of budget for tiny data (<2K rows),
    # ~100% for moderate+ data (>10K rows with many features)
    row_factor = min(1.0, 0.5 + 0.5 * (n_rows / 20000))
    # Many features slow down tree training
    feat_factor = min(1.0, 0.85 + 0.15 * (n_features / 500))
    return min(1.0, row_factor * feat_factor)


def _preset_factor(presets: str) -> float:
    """How presets affect actual time usage relative to budget.

    best_quality uses full budget (stacking, bagging, repeated k-fold).
    good often finishes early.
    """
    factors = {
        "medium_quality": 0.5,
        "good": 0.7,
        "high_quality": 0.85,
        "best_quality": 1.0,
    }
    return factors.get(presets, 0.7)


def _overhead_minutes(n_rows: int, n_features: int, formulation: str) -> float:
    """Non-training overhead: data loading, feature preprocessing, metrics."""
    # Base overhead: load CSV, compute metrics, save artifacts
    base = 0.5
    # Large datasets take longer to load/preprocess
    data_overhead = (n_rows * n_features) / 50_000_000  # ~1 min per 50M cells
    # Multi-label has per-label metric computation
    return base + data_overhead


def _feature_importance_minutes(
    n_features: int, n_labels: int, n_test_rows: int, presets: str, formulation: str
) -> float:
    """Estimate time for permutation-based feature importance.

    train.py calls predictor.feature_importance() per label after training.
    AutoGluon default: num_shuffle_sets=5, subsample_size=5000.
    For each shuffle set, it shuffles one feature at a time and calls
    predict_proba on the full subsample. Total predict calls per label =
    n_features × num_shuffle_sets.

    With best_quality, the best model is WeightedEnsemble_L2 which invokes
    all base models (LightGBM, XGBoost, CatBoost, RF, ExtraTrees) per
    predict call — making it 5-10x slower than a single model.

    Empirical calibration (443 features, 9 labels, 5000 subsample):
    - best_quality: ~8-12 min per label for feature importance
    - good: ~1-2 min per label
    """
    num_shuffle_sets = 5  # AutoGluon default when time_limit=None
    subsample = min(n_test_rows, 5000)  # AG default subsample_size

    # Seconds per predict_proba call on subsample, by model complexity
    # best_quality = stacked ensemble (all base models), good = single LightGBM winner
    predict_seconds_per_call = {
        "medium_quality": 0.02,
        "good": 0.03,
        "high_quality": 0.08,
        "best_quality": 0.20,
    }
    secs_per_call = predict_seconds_per_call.get(presets, 0.03)
    # Scale with subsample size (linear)
    secs_per_call *= subsample / 5000

    # Total calls per label = n_features × num_shuffle_sets
    fi_seconds_per_label = n_features * num_shuffle_sets * secs_per_call

    # Multi-label: done for each label
    n_models = n_labels if formulation == "multilabel" else 1
    total_seconds = fi_seconds_per_label * n_models

    return total_seconds / 60


def _sagemaker_setup_minutes(instance_type: str) -> dict:
    """SageMaker container provisioning and pip install time."""
    return {
        "provisioning": 2.0,  # instance spin-up + image pull
        "pip_install": 4.0,   # autogluon + ray + dependencies (~500MB)
    }


def _format_description(
    formulation: str,
    n_labels: int,
    time_limit: int,
    presets: str,
    execution: str,
    breakdown: dict,
) -> str:
    """Human-readable summary of the estimate."""
    parts = []
    if formulation == "multilabel":
        parts.append(
            f"{n_labels} labels × ~{breakdown['per_label_seconds']}s/label "
            f"(time_limit={time_limit}s, presets={presets})"
        )
    else:
        parts.append(f"time_limit={time_limit}s, presets={presets}")

    parts.append(f"Training: ~{breakdown['model_training']} min")
    parts.append(f"Feature importance: ~{breakdown['feature_importance']} min")
    parts.append(f"Overhead: ~{breakdown['overhead']} min")

    if execution == "sagemaker":
        parts.append(
            f"SageMaker setup: ~{breakdown['container_setup'] + breakdown['pip_install']} min"
        )

    parts.append(f"**Total: ~{breakdown['total_minutes']} min**")
    return " | ".join(parts)


def format_retraining_options(
    n_labels: int = 1,
    n_train_rows: int = 10000,
    n_features: int = 100,
    formulation: str = "multilabel",
    execution: str = "local",
    instance_type: str = "ml.m5.2xlarge",
) -> str:
    """Generate the 3 retraining options with time estimates for experiments.md.

    Returns markdown-formatted options ready to paste into the questionnaire.
    """
    # Option B: Recommended default (--time-limit 120 --presets good --skip-importance)
    recommended = estimate_training_minutes(
        formulation=formulation,
        time_limit=120,
        presets="good",
        n_labels=n_labels,
        n_train_rows=n_train_rows,
        n_features=n_features,
        execution=execution,
        instance_type=instance_type,
        skip_importance=True,
    )
    # Option C: Full budget (--time-limit 300 --presets best_quality --skip-importance)
    # Only for large datasets where stacking reliably helps
    full = estimate_training_minutes(
        formulation=formulation,
        time_limit=300,
        presets="best_quality",
        n_labels=n_labels,
        n_train_rows=n_train_rows,
        n_features=n_features,
        execution=execution,
        instance_type=instance_type,
        skip_importance=True,
    )

    # Warning for small datasets
    warning = ""
    if n_train_rows < 50000:
        warning = (
            "\n> ⚠️ **Warning**: With <50K training rows, `best_quality` often overfits "
            "(stacked ensembles hurt). Option B is recommended.\n"
        )

    lines = [
        "### Final Model Retraining",
        "",
        "The best experiment has been selected. Choose the final training configuration:",
        "",
        "| Option | Configuration | Estimated Time | When to use |",
        "|--------|--------------|----------------|-------------|",
        f"| A) **Skip** (recommended) | Use best experiment model as-is | 0 min | Results already meet requirements; feature importance already computed during experiments |",
        f"| B) **Extended** | `--time-limit 120 --presets good --skip-importance` | ~{recommended['total_minutes']} min | Want more model iterations without overfitting risk |",
        f"| C) **Full** | `--time-limit 300 --presets best_quality --skip-importance` | ~{full['total_minutes']} min | Only for datasets >50K rows where stacking reliably helps |",
        warning,
        "[Answer]:",
        "",
    ]
    return "\n".join(lines)
