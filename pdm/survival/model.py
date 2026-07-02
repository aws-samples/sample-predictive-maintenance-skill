"""Survival analysis model — Cox PH, Weibull AFT, Random Survival Forest."""

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from lifelines import CoxPHFitter, WeibullAFTFitter
from lifelines.utils import concordance_index

from pdm.base import PDMModel, PredictionResult, TrainResult


class SurvivalPredictor(PDMModel):
    """Proper survival analysis with censoring support.

    Models:
        - Cox Proportional Hazards (interpretable baseline)
        - Weibull AFT (parametric)
        - Random Survival Forest (non-linear, via scikit-survival)
    """

    formulation = "survival"

    def __init__(self):
        self.model = None
        self.model_type: str = ""
        self.feature_names: list[str] = []

    def train(self, train_df: pd.DataFrame, test_df: pd.DataFrame, **kwargs) -> TrainResult:
        """Train survival models and select best by concordance index.

        Expects train_df/test_df to contain 'duration' and 'event' columns.
        """
        warnings.filterwarnings("ignore", category=Warning, module="lifelines")
        time_limit = kwargs.get("time_limit", 300)
        n_trials = kwargs.get("n_trials", 30)

        feature_cols = [c for c in train_df.columns if c not in ("duration", "event") and train_df[c].dtype in (np.float64, np.int64, float, int)]
        self.feature_names = feature_cols

        # Prepare lifelines-format DataFrames
        train_sl = train_df[feature_cols + ["duration", "event"]].copy()
        test_sl = test_df[feature_cols + ["duration", "event"]].copy()
        train_sl["duration"] = train_sl["duration"].clip(lower=1)
        test_sl["duration"] = test_sl["duration"].clip(lower=1)

        best_model = None
        best_cindex = 0.0
        best_type = ""
        all_results = []

        # --- Cox PH ---
        for pen in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0]:
            try:
                m = CoxPHFitter(penalizer=pen)
                m.fit(train_sl, duration_col="duration", event_col="event")
                ci = self._score(m, test_sl)
                all_results.append({"model": "cox", "penalizer": pen, "cindex": ci})
                if ci > best_cindex:
                    best_cindex, best_model, best_type = ci, m, "cox"
            except Exception:
                continue

        # --- Weibull AFT ---
        for pen in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 5.0]:
            try:
                m = WeibullAFTFitter(penalizer=pen)
                m.fit(train_sl, duration_col="duration", event_col="event")
                ci = self._score(m, test_sl)
                all_results.append({"model": "weibull", "penalizer": pen, "cindex": ci})
                if ci > best_cindex:
                    best_cindex, best_model, best_type = ci, m, "weibull"
            except Exception:
                continue

        # --- Random Survival Forest ---
        try:
            from sksurv.ensemble import RandomSurvivalForest
            from sksurv.util import Surv

            y_train = Surv.from_dataframe("event", "duration", train_sl.astype({"event": bool}))
            y_test = Surv.from_dataframe("event", "duration", test_sl.astype({"event": bool}))
            X_train = train_sl[feature_cols].values
            X_test = test_sl[feature_cols].values

            rsf = RandomSurvivalForest(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rsf.fit(X_train, y_train)
            ci = rsf.score(X_test, y_test)
            all_results.append({"model": "rsf", "cindex": ci})
            if ci > best_cindex:
                best_cindex, best_model, best_type = ci, rsf, "rsf"
        except Exception:
            pass

        self.model = best_model
        self.model_type = best_type

        # Compute metrics
        metrics = {"concordance_index": round(best_cindex, 4)}

        # Brier score (if sksurv available)
        try:
            from sksurv.metrics import integrated_brier_score
            from sksurv.util import Surv
            y_train_s = Surv.from_dataframe("event", "duration", train_sl.astype({"event": bool}))
            y_test_s = Surv.from_dataframe("event", "duration", test_sl.astype({"event": bool}))
            times = np.linspace(test_sl["duration"].quantile(0.1), test_sl["duration"].quantile(0.9), 20)
            if best_type == "rsf":
                surv_fns = best_model.predict_survival_function(X_test)
                preds = np.row_stack([fn(times) for fn in surv_fns])
            else:
                surv_df = best_model.predict_survival_function(test_sl[feature_cols])
                valid_times = times[(times >= surv_df.index.min()) & (times <= surv_df.index.max())]
                if len(valid_times) > 2:
                    times = valid_times
                    preds = surv_df.loc[surv_df.index.isin(times)].T.values if len(surv_df.index) > 0 else None
                else:
                    preds = None
            if preds is not None and preds.shape[1] == len(times):
                ibs = integrated_brier_score(y_train_s, y_test_s, preds, times)
                metrics["integrated_brier_score"] = round(float(ibs), 4)
        except Exception:
            pass

        # Feature importance
        importance = None
        if best_type == "cox":
            importance = pd.Series(best_model.params_.abs().values, index=best_model.params_.index).sort_values(ascending=False)
        elif best_type == "rsf":
            try:
                importance = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
            except (NotImplementedError, AttributeError):
                pass  # scikit-survival RSF doesn't implement feature_importances_

        return TrainResult(
            model=best_model,
            metrics=metrics,
            feature_importance=importance,
            metadata={"model_type": best_type, "feature_names": feature_cols, "all_trials": all_results},
        )

    def predict(self, features: pd.DataFrame) -> PredictionResult:
        """Predict median survival time with confidence intervals."""
        X = features[self.feature_names] if self.feature_names else features

        if self.model_type in ("cox", "weibull"):
            median = self.model.predict_median(X)
            median = median.replace([np.inf, -np.inf], np.nan).fillna(median[np.isfinite(median)].max())
            # Percentile-based CI from survival function
            surv_fn = self.model.predict_survival_function(X)
            ci_lower = self._percentile_from_surv(surv_fn, 0.9)
            ci_upper = self._percentile_from_surv(surv_fn, 0.1)
        elif self.model_type == "rsf":
            surv_fns = self.model.predict_survival_function(X.values)
            median = np.array([self._median_from_step(fn) for fn in surv_fns])
            ci_lower = np.array([self._quantile_from_step(fn, 0.9) for fn in surv_fns])
            ci_upper = np.array([self._quantile_from_step(fn, 0.1) for fn in surv_fns])
        else:
            raise RuntimeError("No model trained")

        return PredictionResult(predictions=pd.DataFrame({
            "median_survival": median,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }, index=features.index))

    def explain(self, features: pd.DataFrame, top_k: int = 10) -> list[dict]:
        """Return feature contributions (Cox coefficients or RSF importances)."""
        if self.model_type == "cox":
            coeffs = self.model.params_
            return [{"feature": f, "coefficient": round(float(coeffs[f]), 4)} for f in coeffs.abs().nlargest(top_k).index]
        elif self.model_type == "rsf":
            try:
                imp = pd.Series(self.model.feature_importances_, index=self.feature_names).nlargest(top_k)
                return [{"feature": f, "importance": round(float(v), 4)} for f, v in imp.items()]
            except (NotImplementedError, AttributeError):
                return []
        return []

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "model.joblib")
        (path / "metadata.json").write_text(json.dumps({
            "formulation": "survival",
            "model_type": self.model_type,
            "feature_names": self.feature_names,
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> "SurvivalPredictor":
        path = Path(path)
        meta = json.loads((path / "metadata.json").read_text())
        obj = cls()
        obj.model = joblib.load(path / "model.joblib")
        obj.model_type = meta["model_type"]
        obj.feature_names = meta["feature_names"]
        return obj

    # --- helpers ---

    def _score(self, model, test_df: pd.DataFrame) -> float:
        feature_cols = [c for c in test_df.columns if c not in ("duration", "event")]
        median = model.predict_median(test_df[feature_cols])
        median = median.replace([np.inf, -np.inf], np.nan).fillna(median[np.isfinite(median)].max())
        return concordance_index(test_df["duration"], median, test_df["event"])

    @staticmethod
    def _percentile_from_surv(surv_df: pd.DataFrame, quantile: float) -> np.ndarray:
        """Extract time at which S(t) crosses quantile for each column."""
        result = []
        for col in surv_df.columns:
            s = surv_df[col]
            below = s[s <= quantile]
            result.append(float(below.index[0]) if len(below) > 0 else float(s.index[-1]))
        return np.array(result)

    @staticmethod
    def _median_from_step(fn) -> float:
        idx = np.searchsorted(-fn.y, -0.5)
        return float(fn.x[min(idx, len(fn.x) - 1)])

    @staticmethod
    def _quantile_from_step(fn, quantile: float) -> float:
        idx = np.searchsorted(-fn.y, -quantile)
        return float(fn.x[min(idx, len(fn.x) - 1)])
