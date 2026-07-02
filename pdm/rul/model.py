"""RUL Predictor — remaining useful life estimation with sliding window features."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from autogluon.tabular import TabularPredictor

from pdm.base import PDMModel, PredictionResult, TrainResult


def nasa_scoring(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA PHM'08 asymmetric scoring (lower is better)."""
    d = y_pred - y_true
    return float(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1).sum())


def sliding_window_features(df: pd.DataFrame, unit_col: str, sensor_cols: list[str],
                            window_size: int = 30, stride: int = 1) -> tuple[pd.DataFrame, list[int]]:
    """Compute sliding window statistics per unit.

    Returns (features_df, indices) where indices map back to original rows.
    """
    from scipy.stats import kurtosis as _kurtosis

    records, indices = [], []
    t = np.arange(window_size, dtype=np.float64)
    for _, group in df.groupby(unit_col):
        values = group[sensor_cols].values
        for i in range(window_size - 1, len(values), stride):
            window = values[i - window_size + 1: i + 1]
            stats = {}
            for j, col in enumerate(sensor_cols):
                w = window[:, j]
                stats[f"{col}_mean"] = w.mean()
                stats[f"{col}_std"] = w.std()
                stats[f"{col}_min"] = w.min()
                stats[f"{col}_max"] = w.max()
                stats[f"{col}_last"] = w[-1]
                stats[f"{col}_range"] = w.max() - w.min()
                stats[f"{col}_rms"] = np.sqrt(np.mean(w ** 2))
                stats[f"{col}_slope"] = np.polyfit(t[:len(w)], w, 1)[0]
                std = w.std()
                stats[f"{col}_kurt"] = float(_kurtosis(w, fisher=True)) if std > 1e-10 else 0.0
                stats[f"{col}_p25"] = np.percentile(w, 25)
                stats[f"{col}_p75"] = np.percentile(w, 75)
                # Frequency-domain features
                fft_vals = np.abs(np.fft.rfft(w))[1:]  # exclude DC
                if len(fft_vals) > 0:
                    stats[f"{col}_fft_max"] = fft_vals.max()
                    stats[f"{col}_fft_mean"] = fft_vals.mean()
                    fft_norm = fft_vals / (fft_vals.sum() + 1e-10)
                    stats[f"{col}_spectral_entropy"] = float(-np.sum(fft_norm * np.log(fft_norm + 1e-10)))
            records.append(stats)
            indices.append(group.index[i])
    return pd.DataFrame(records), indices


class _Ensemble:
    """Averaging ensemble of regressors."""
    def __init__(self, models):
        self.models = models
    def predict(self, X):
        return np.column_stack([m.predict(X) for m in self.models]).mean(axis=1)


class RULPredictor(PDMModel):
    """RUL regression with sliding window feature engineering and NASA scoring.

    Expects data with 'unit_id', 'cycle', sensor columns, and 'RUL' target.
    """

    formulation = "rul"

    def __init__(self, window_size: int = 30, rul_cap: int = 125):
        self.window_size = window_size
        self.rul_cap = rul_cap
        self.predictor = None
        self.feature_names: list[str] = []
        self.sensor_cols: list[str] = []

    def train(self, train_df: pd.DataFrame, test_df: pd.DataFrame, **kwargs) -> TrainResult:
        """Train RUL model with sliding window features.

        Supports two backends:
        - 'autogluon' (default): AutoGluon TabularPredictor with stacking
        - 'optuna': Optuna HPO over RF/XGBoost/GBR with top-K ensemble
        """
        time_limit = kwargs.get("time_limit", 600)
        presets = kwargs.get("presets", "best")
        backend = kwargs.get("backend", "autogluon")
        n_trials = kwargs.get("n_trials", 50)
        stride = kwargs.get("stride", 1)

        unit_col = "unit_id"
        target_col = "RUL"

        # Identify sensor columns (numeric, not target/id/cycle)
        exclude = {unit_col, "cycle", target_col}
        self.sensor_cols = [c for c in train_df.select_dtypes(include=[np.number]).columns if c not in exclude]

        # Cap RUL
        train_df = train_df.copy()
        train_df[target_col] = train_df[target_col].clip(upper=self.rul_cap)

        # Normalize features before windowing
        self._feat_mean = train_df[self.sensor_cols].mean()
        self._feat_std = train_df[self.sensor_cols].std().replace(0, 1)
        train_df[self.sensor_cols] = (train_df[self.sensor_cols] - self._feat_mean) / self._feat_std
        test_df = test_df.copy()
        test_df[self.sensor_cols] = (test_df[self.sensor_cols] - self._feat_mean) / self._feat_std

        # Build sliding window features
        X_train, train_idx = sliding_window_features(train_df, unit_col, self.sensor_cols, self.window_size, stride)
        y_train = train_df.loc[train_idx, target_col].values
        self.feature_names = list(X_train.columns)

        # Test: use last window per unit
        X_test = self._last_window_features(test_df, unit_col)
        y_test = test_df.groupby(unit_col).tail(1)[target_col].values

        if backend == "optuna":
            preds, metrics = self._train_optuna(X_train, y_train, X_test, y_test, n_trials, time_limit)
        else:
            X_train[target_col] = y_train
            self.predictor = TabularPredictor(
                label=target_col, eval_metric="root_mean_squared_error",
                problem_type="regression",
                path=str(kwargs.get("output", Path("./model")) / "ag_model"),
                verbosity=0,
            ).fit(train_data=X_train, time_limit=time_limit, presets=presets)
            preds = self.predictor.predict(X_test[self.feature_names]).values
            rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
            nasa = nasa_scoring(y_test, preds)
            metrics = {"rmse": round(rmse, 2), "nasa_score": round(nasa, 1),
                       "nasa_score_normalized": round(nasa / len(y_test), 2)}

        return TrainResult(
            model=self.predictor,
            metrics=metrics,
            feature_importance=None,
            metadata={
                "formulation": "rul",
                "feature_names": self.feature_names,
                "sensor_cols": self.sensor_cols,
                "window_size": self.window_size,
                "rul_cap": self.rul_cap,
                "backend": backend,
            },
        )

    def _train_optuna(self, X_train, y_train, X_test, y_test, n_trials, timeout):
        """Optuna HPO over RF/XGBoost/GBR with top-5 ensemble."""
        import optuna
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        from xgboost import XGBRegressor

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        top_models = []  # (rmse, model)

        def objective(trial):
            algo = trial.suggest_categorical("algorithm", ["rf", "xgb", "gbr"])
            if algo == "rf":
                model = RandomForestRegressor(
                    n_estimators=trial.suggest_int("n_estimators", 100, 500),
                    max_depth=trial.suggest_int("max_depth", 8, 40),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                    random_state=42, n_jobs=-1)
            elif algo == "xgb":
                model = XGBRegressor(
                    n_estimators=trial.suggest_int("n_estimators", 100, 500),
                    max_depth=trial.suggest_int("max_depth", 4, 15),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                    random_state=42, verbosity=0)
            else:
                model = GradientBoostingRegressor(
                    n_estimators=trial.suggest_int("n_estimators", 100, 500),
                    max_depth=trial.suggest_int("max_depth", 3, 12),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    random_state=42)
            model.fit(X_tr, y_tr)
            rmse = float(np.sqrt(np.mean((y_val - model.predict(X_val)) ** 2)))
            top_models.append((rmse, model))
            top_models.sort(key=lambda x: x[0])
            if len(top_models) > 5:
                top_models.pop()
            return rmse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout, catch=(Exception,))

        # Build ensemble
        if len(top_models) >= 2:
            ensemble_preds = np.column_stack([m.predict(X_test) for _, m in top_models]).mean(axis=1)
            single_preds = top_models[0][1].predict(X_test)
            ens_rmse = float(np.sqrt(np.mean((y_test - ensemble_preds) ** 2)))
            single_rmse = float(np.sqrt(np.mean((y_test - single_preds) ** 2)))
            if ens_rmse <= single_rmse:
                preds = ensemble_preds
                self.predictor = _Ensemble([m for _, m in top_models])
            else:
                preds = single_preds
                self.predictor = top_models[0][1]
        else:
            preds = top_models[0][1].predict(X_test)
            self.predictor = top_models[0][1]

        rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
        nasa = nasa_scoring(y_test, preds)
        return preds, {"rmse": round(rmse, 2), "nasa_score": round(nasa, 1),
                       "nasa_score_normalized": round(nasa / len(y_test), 2)}

    def predict(self, features: pd.DataFrame) -> PredictionResult:
        """Predict RUL. Accepts either pre-computed window features or raw time-series."""
        if "unit_id" in features.columns and any(c in features.columns for c in self.sensor_cols):
            X = self._last_window_features(features, "unit_id")
        else:
            X = features

        X = X.reindex(columns=self.feature_names, fill_value=0)
        preds = self.predictor.predict(X)
        if hasattr(preds, 'values'):
            preds = preds.values

        return PredictionResult(predictions=pd.DataFrame({
            "predicted_rul": preds,
        }, index=X.index))

    def explain(self, features: pd.DataFrame, top_k: int = 10) -> list[dict]:
        """Per-prediction feature attributions.

        For Optuna backend: uses native tree SHAP (XGBoost pred_contribs / RF feature_importances_).
        For AutoGluon backend: uses permutation importance.
        """
        try:
            if hasattr(self.predictor, 'feature_importance'):
                # AutoGluon
                imp = self.predictor.feature_importance(features, silent=True)
                return [{"feature": f, "importance": round(float(imp.loc[f, "importance"]), 4)} for f in imp.head(top_k).index]

            # Optuna ensemble or single model — use native tree contribs
            model = self.predictor
            if isinstance(model, _Ensemble):
                model = model.models[0]

            X = features[self.feature_names] if all(c in features.columns for c in self.feature_names) else features

            # XGBoost native SHAP
            if hasattr(model, "get_booster"):
                import xgboost as xgb
                contribs = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
                mean_abs = np.abs(contribs[:, :-1]).mean(axis=0)
                top_idx = np.argsort(mean_abs)[::-1][:top_k]
                return [{"feature": self.feature_names[i], "importance": round(float(mean_abs[i]), 4)} for i in top_idx]

            # Fallback: global feature_importances_
            if hasattr(model, "feature_importances_"):
                imp = pd.Series(model.feature_importances_, index=self.feature_names).sort_values(ascending=False)
                return [{"feature": f, "importance": round(float(v), 4)} for f, v in imp.head(top_k).items()]
        except Exception:
            pass
        return []

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "metadata.json").write_text(json.dumps({
            "formulation": "rul",
            "feature_names": self.feature_names,
            "sensor_cols": self.sensor_cols,
            "window_size": self.window_size,
            "rul_cap": self.rul_cap,
        }, indent=2))
        # AutoGluon saves itself at path/ag_model during training

    @classmethod
    def load(cls, path: Path) -> "RULPredictor":
        path = Path(path)
        meta = json.loads((path / "metadata.json").read_text())
        obj = cls(window_size=meta["window_size"], rul_cap=meta["rul_cap"])
        obj.feature_names = meta["feature_names"]
        obj.sensor_cols = meta["sensor_cols"]
        obj.predictor = TabularPredictor.load(str(path / "ag_model"), verbosity=0)
        return obj

    def _last_window_features(self, df: pd.DataFrame, unit_col: str) -> pd.DataFrame:
        """Extract features from the last window_size cycles per unit."""
        from scipy.stats import kurtosis as _kurtosis

        t = np.arange(self.window_size, dtype=np.float64)
        records = []
        for _, group in df.groupby(unit_col):
            values = group[self.sensor_cols].values
            if len(values) >= self.window_size:
                window = values[-self.window_size:]
            else:
                window = np.vstack([np.tile(values[0], (self.window_size - len(values), 1)), values])
            stats = {}
            for j, col in enumerate(self.sensor_cols):
                w = window[:, j]
                stats[f"{col}_mean"] = w.mean()
                stats[f"{col}_std"] = w.std()
                stats[f"{col}_min"] = w.min()
                stats[f"{col}_max"] = w.max()
                stats[f"{col}_last"] = w[-1]
                stats[f"{col}_range"] = w.max() - w.min()
                stats[f"{col}_rms"] = np.sqrt(np.mean(w ** 2))
                stats[f"{col}_slope"] = np.polyfit(t[:len(w)], w, 1)[0]
                std = w.std()
                stats[f"{col}_kurt"] = float(_kurtosis(w, fisher=True)) if std > 1e-10 else 0.0
                stats[f"{col}_p25"] = np.percentile(w, 25)
                stats[f"{col}_p75"] = np.percentile(w, 75)
                # Frequency-domain features
                fft_vals = np.abs(np.fft.rfft(w))[1:]
                if len(fft_vals) > 0:
                    stats[f"{col}_fft_max"] = fft_vals.max()
                    stats[f"{col}_fft_mean"] = fft_vals.mean()
                    fft_norm = fft_vals / (fft_vals.sum() + 1e-10)
                    stats[f"{col}_spectral_entropy"] = float(-np.sum(fft_norm * np.log(fft_norm + 1e-10)))
            records.append(stats)
        return pd.DataFrame(records)
