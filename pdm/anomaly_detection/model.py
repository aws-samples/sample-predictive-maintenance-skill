"""Anomaly Detection model — Isolation Forest wrapped in PDMModel interface."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from pdm.base import PDMModel, PredictionResult, TrainResult


class AnomalyDetector(PDMModel):
    """Unsupervised anomaly detection via Isolation Forest."""

    formulation = "anomaly_detection"

    def __init__(self, contamination: float = 0.05, n_estimators: int = 200):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model: IsolationForest | None = None
        self.scaler: StandardScaler | None = None
        self.threshold: float = 0.0
        self.feature_names: list[str] = []

    def train(self, train_df: pd.DataFrame, test_df: pd.DataFrame, **kwargs) -> TrainResult:
        """Train Isolation Forest on normal-only data."""
        feature_cols = self._select_features(train_df)
        normal_mask = self._detect_normal_mask(train_df)
        train_normal = train_df.loc[normal_mask, feature_cols].fillna(0)

        # Drop zero-variance
        var = train_normal.var()
        feature_cols = [c for c in feature_cols if var.get(c, 0) > 0]
        train_normal = train_normal[feature_cols]
        self.feature_names = feature_cols

        # Scale and train
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(train_normal)

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=42, n_jobs=-1,
        )
        self.model.fit(X_train)

        # Threshold from training scores
        train_scores = -self.model.score_samples(X_train)
        self.threshold = float(np.percentile(train_scores, (1 - self.contamination) * 100))

        # Evaluate on test
        X_test = self.scaler.transform(test_df[feature_cols].fillna(0))
        test_scores = -self.model.score_samples(X_test)
        test_anomaly_rate = float((test_scores > self.threshold).mean())

        metrics = {
            "threshold": self.threshold,
            "train_anomaly_rate": float((train_scores > self.threshold).mean()),
            "test_anomaly_rate": test_anomaly_rate,
        }

        return TrainResult(
            model=self.model,
            metrics=metrics,
            metadata={"formulation": "anomaly_detection", "feature_names": feature_cols,
                      "contamination": self.contamination, "n_estimators": self.n_estimators},
        )

    def predict(self, features: pd.DataFrame) -> PredictionResult:
        """Score samples and flag anomalies."""
        X = self.scaler.transform(features[self.feature_names].fillna(0))
        scores = -self.model.score_samples(X)
        return PredictionResult(predictions=pd.DataFrame({
            "anomaly_score": scores,
            "is_anomaly": (scores > self.threshold).astype(int),
        }, index=features.index))

    def explain(self, features: pd.DataFrame, top_k: int = 10) -> list[dict]:
        """Per-sample feature contribution via perturbation."""
        results = []
        X = self.scaler.transform(features[self.feature_names].fillna(0))
        for i in range(len(X)):
            base_score = -self.model.score_samples(X[i:i + 1])[0]
            contribs = {}
            for j, fname in enumerate(self.feature_names):
                perturbed = X[i].copy()
                perturbed[j] = 0.0
                contribs[fname] = float(base_score - (-self.model.score_samples(perturbed.reshape(1, -1))[0]))
            top = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
            results.append([{"feature": f, "contribution": round(v, 6)} for f, v in top])
        return results

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "isolation_forest.joblib")
        joblib.dump(self.scaler, path / "scaler.joblib")
        (path / "metadata.json").write_text(json.dumps({
            "formulation": "anomaly_detection",
            "feature_names": self.feature_names,
            "threshold": self.threshold,
            "contamination": self.contamination,
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> "AnomalyDetector":
        path = Path(path)
        meta = json.loads((path / "metadata.json").read_text())
        obj = cls(contamination=meta.get("contamination", 0.05))
        obj.model = joblib.load(path / "isolation_forest.joblib")
        obj.scaler = joblib.load(path / "scaler.joblib")
        obj.feature_names = meta["feature_names"]
        obj.threshold = meta["threshold"]
        return obj

    @staticmethod
    def _select_features(df: pd.DataFrame) -> list[str]:
        exclude = {"RUL", "machine_failure", "duration", "event", "unit_id", "device_id", "observation_date"}
        return [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in exclude and not c.startswith("label_")]

    @staticmethod
    def _detect_normal_mask(df: pd.DataFrame) -> pd.Series:
        label_cols = [c for c in df.columns if c.startswith("label_")]
        if label_cols:
            return df[label_cols].sum(axis=1) == 0
        if "machine_failure" in df.columns:
            return df["machine_failure"] == 0
        return pd.Series(True, index=df.index)
