"""Multiple anomaly detection algorithms via PyOD.

Provides a registry of algorithms and auto-selection via synthetic injection test.

Usage:
    from pdm.anomaly_detection.algorithms import create_detector, auto_select, ADAlgorithm

    # Specific algorithm
    detector = create_detector(ADAlgorithm.COPOD, contamination=0.05)
    detector.fit(X_train)
    scores = detector.decision_function(X_test)

    # Auto-select best
    best_algo, results = auto_select(train_df, feature_cols)
"""
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd


class ADAlgorithm(Enum):
    ISOLATION_FOREST = "isolation_forest"
    LOF = "lof"
    OCSVM = "ocsvm"
    COPOD = "copod"
    ECOD = "ecod"


def create_detector(algorithm: ADAlgorithm, contamination: float = 0.05,
                    n_features: int = 10, **kwargs) -> Any:
    """Create a PyOD-compatible detector.

    All detectors expose: .fit(X), .decision_function(X), .labels_

    Args:
        algorithm: Which algorithm to use
        contamination: Expected fraction of anomalies
        n_features: Number of input features (used for autoencoder sizing)

    Returns:
        PyOD detector instance (unfitted)
    """
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.models.ocsvm import OCSVM
    from pyod.models.copod import COPOD
    from pyod.models.ecod import ECOD

    if algorithm == ADAlgorithm.ISOLATION_FOREST:
        return IForest(contamination=contamination,
                      n_estimators=kwargs.get("n_estimators", 200),
                      random_state=42, n_jobs=-1)

    elif algorithm == ADAlgorithm.LOF:
        return LOF(contamination=contamination,
                  n_neighbors=kwargs.get("n_neighbors", 20),
                  novelty=True, n_jobs=-1)

    elif algorithm == ADAlgorithm.OCSVM:
        return OCSVM(contamination=contamination, kernel="rbf")

    elif algorithm == ADAlgorithm.COPOD:
        return COPOD(contamination=contamination)

    elif algorithm == ADAlgorithm.ECOD:
        return ECOD(contamination=contamination)

    raise ValueError(f"Unknown algorithm: {algorithm}")


def auto_select(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    contamination: float = 0.05,
    candidates: Optional[list[ADAlgorithm]] = None,
    n_trials: int = 3,
) -> tuple[ADAlgorithm, dict]:
    """Auto-select best AD algorithm via synthetic anomaly injection.

    Strategy:
        1. Train each candidate on (assumed normal) training data
        2. Inject synthetic anomalies (spikes)
        3. Measure detection rate
        4. Pick the algorithm with highest mean detection rate

    Args:
        train_df: Training data (assumed mostly normal)
        feature_cols: Numeric feature columns
        contamination: Expected anomaly fraction
        candidates: Algorithms to try (default: IF, COPOD, ECOD, LOF)
        n_trials: Number of random injection trials

    Returns:
        (best_algorithm, results_dict) where results maps algorithm to stats
    """
    from sklearn.preprocessing import StandardScaler

    if candidates is None:
        candidates = [ADAlgorithm.ISOLATION_FOREST, ADAlgorithm.COPOD,
                      ADAlgorithm.ECOD, ADAlgorithm.LOF]

    X = StandardScaler().fit_transform(train_df[feature_cols].fillna(0).values)

    results = {}
    for algo in candidates:
        detection_rates = []
        for trial in range(n_trials):
            try:
                detector = create_detector(algo, contamination=contamination)
                detector.fit(X)

                # Inject spike anomalies into a copy
                rng = np.random.default_rng(42 + trial)
                n_test = min(200, len(X))
                X_test = X[:n_test].copy()
                n_inject = max(5, int(n_test * 0.1))
                inject_idx = rng.choice(n_test, n_inject, replace=False)
                X_test[inject_idx] += rng.normal(0, 3, (n_inject, X_test.shape[1]))

                # Score and detect
                scores = detector.decision_function(X_test)
                threshold = np.percentile(scores, (1 - contamination) * 100)
                predicted = (scores > threshold).astype(int)

                # Detection rate on injected anomalies
                detection_rate = float(predicted[inject_idx].mean())
                detection_rates.append(detection_rate)
            except Exception:
                continue

        if detection_rates:
            results[algo] = {
                "mean_detection_rate": float(np.mean(detection_rates)),
                "std_detection_rate": float(np.std(detection_rates)),
                "n_trials": len(detection_rates),
            }

    if not results:
        return ADAlgorithm.ISOLATION_FOREST, {"fallback": True}

    best = max(results, key=lambda k: results[k]["mean_detection_rate"])
    return best, {k.value: v for k, v in results.items()}
