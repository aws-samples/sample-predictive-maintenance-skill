#!/usr/bin/env python3
"""Inference script for PdM model (AutoGluon).

Usage:
    uv run python pdm/fault_prediction/inference.py                    # First 5 test samples
    uv run python pdm/fault_prediction/inference.py -n 10              # First 10 test samples
    uv run python pdm/fault_prediction/inference.py --input data.csv   # Custom input file
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor


class PDMModel:
    """Predictive Maintenance model wrapper with debug capabilities."""

    def __init__(self, model_path: str = "./model"):
        model_dir = Path(model_path)
        with open(model_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        self.feature_names = self.metadata["feature_names"]
        self.formulation = self.metadata["formulation"]

        ag_model_path = model_dir / "ag_model"
        if "label_names" in self.metadata:
            self.label_names = self.metadata["label_names"]
            self.predictors = {}
            for label in self.label_names:
                label_dir = ag_model_path / label.replace(" ", "_")
                if label_dir.exists():
                    self.predictors[label] = TabularPredictor.load(str(label_dir), verbosity=0)
        else:
            self.label_names = [self.metadata.get("target_name", "target")]
            self.predictor = TabularPredictor.load(str(ag_model_path), verbosity=0)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X = features.reindex(columns=self.feature_names, fill_value=0)

        if hasattr(self, "predictors"):
            from concurrent.futures import ThreadPoolExecutor

            def _predict_label(label_predictor):
                label, predictor = label_predictor
                pred = predictor.predict(X).values
                try:
                    proba = predictor.predict_proba(X).iloc[:, 1].values
                except Exception:
                    proba = None
                return label, pred, proba

            results = {}
            with ThreadPoolExecutor(max_workers=len(self.predictors)) as executor:
                for label, pred, proba in executor.map(_predict_label, self.predictors.items()):
                    results[f"{label}_pred"] = pred
                    if proba is not None:
                        results[f"{label}_proba"] = proba
            return pd.DataFrame(results, index=features.index)
        else:
            preds = self.predictor.predict(X)
            result = {"prediction": preds.values}
            if self.formulation in ("classification", "binary"):
                try:
                    proba = self.predictor.predict_proba(X)
                    result["proba"] = proba.iloc[:, 1].values
                except Exception:
                    pass
            return pd.DataFrame(result, index=features.index)

    def predict_debug(self, features: pd.DataFrame, labels: pd.DataFrame | None = None,
                      top_features: int = 10) -> list[dict]:
        predictions = self.predict(features)
        debug_output = []

        for i in range(len(features)):
            row = features.iloc[i]
            non_zero = row[row != 0].abs().sort_values(ascending=False)
            top = non_zero.head(top_features)

            entry = {
                "sample_index": i,
                "top_input_features": {name: round(float(row[name]), 4) for name in top.index},
                "predictions": {},
            }

            if hasattr(self, "predictors"):
                for label in self.label_names:
                    pred_col = f"{label}_pred"
                    proba_col = f"{label}_proba"
                    if pred_col in predictions.columns:
                        pred_info = {
                            "predicted": int(predictions.iloc[i][pred_col]),
                            "probability": round(float(predictions.iloc[i].get(proba_col, 0)), 4),
                        }
                        if labels is not None and label in labels.columns:
                            pred_info["actual"] = int(labels.iloc[i][label])
                            pred_info["correct"] = pred_info["predicted"] == pred_info["actual"]
                        entry["predictions"][label] = pred_info
            else:
                pred_info = {"predicted": predictions.iloc[i]["prediction"]}
                if "proba" in predictions.columns:
                    pred_info["probability"] = round(float(predictions.iloc[i]["proba"]), 4)
                if labels is not None:
                    target = self.metadata.get("target_name", "target")
                    if target in labels.columns:
                        pred_info["actual"] = float(labels.iloc[i][target])
                entry["predictions"] = pred_info

            debug_output.append(entry)
        return debug_output


def main():
    parser = argparse.ArgumentParser(description="Run PdM inference with debug output")
    parser.add_argument("-n", type=int, default=5)
    parser.add_argument("--input", type=Path, default=None,
                        help="Input CSV. Default: auto-discovers test.csv relative to --model-path")
    parser.add_argument("--model-path", type=Path, default=Path("./model"))
    parser.add_argument("--top-features", type=int, default=10)
    parser.add_argument("--explain", action="store_true", help="Use SHAP for per-prediction explanations")
    parser.add_argument("--counterfactual", action="store_true", help="Show what feature changes would flip the prediction")
    args = parser.parse_args()

    # Auto-discover input file if not provided
    if args.input is None:
        # Try: sibling data/ folder (experiments/NN/model -> experiments/NN/data/test.csv)
        candidate = args.model_path.parent / "data" / "test.csv"
        if candidate.exists():
            args.input = candidate
        else:
            # Fallback: ./data/raw_test.csv (project root)
            fallback = Path("./data/raw_test.csv")
            if fallback.exists():
                args.input = fallback
            else:
                parser.error("No input file found. Provide --input explicitly.")

    model = PDMModel(str(args.model_path))
    print(f"Model loaded: {model.formulation}")
    print(f"  Features: {len(model.feature_names)} | Labels: {len(model.label_names)}")

    df = pd.read_csv(args.input, nrows=args.n)
    label_cols = [c for c in df.columns if c.startswith("label_") or c == model.metadata.get("target_name")]
    feature_cols = [c for c in df.columns if c not in label_cols]

    features = df[feature_cols].fillna(0)
    labels = df[label_cols] if label_cols else None

    print(f"\nPredicting {len(df)} samples from {args.input}...\n")

    # Per-sample latency measurement (excludes model loading)
    import time
    latencies_ms = []
    for i in range(len(features)):
        t0 = time.perf_counter()
        _ = model.predict(features.iloc[i:i+1])
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    results = model.predict_debug(features, labels=labels, top_features=args.top_features)

    # SHAP explanations
    shap_values_per_sample = None
    if args.explain:
        try:
            import shap
            print("Computing SHAP explanations...\n")
            if hasattr(model, "predictors"):
                # Multi-label: explain the first active predictor
                first_label = next(iter(model.predictors))
                explainer = shap.TreeExplainer(model.predictors[first_label]._trainer.load_model(
                    model.predictors[first_label].model_best))
                shap_values_per_sample = explainer.shap_values(features.reindex(columns=model.feature_names, fill_value=0))
            else:
                explainer = shap.TreeExplainer(model.predictor._trainer.load_model(model.predictor.model_best))
                shap_values_per_sample = explainer.shap_values(features.reindex(columns=model.feature_names, fill_value=0))
        except Exception as e:
            print(f"  ⚠️ SHAP failed ({e}), falling back to raw feature values\n")

    for entry in results:
        print(f"{'─'*70}")
        print(f"Sample {entry['sample_index']}:")
        print("  Top input features:")
        if shap_values_per_sample is not None:
            sv = shap_values_per_sample[entry['sample_index']]
            if hasattr(sv, '__len__') and len(sv) == len(model.feature_names):
                top_idx = np.argsort(np.abs(sv))[::-1][:args.top_features]
                for idx in top_idx:
                    fname = model.feature_names[idx]
                    print(f"    {fname:45s} SHAP={sv[idx]:+.4f}")
            else:
                for name, val in entry["top_input_features"].items():
                    print(f"    {name:45s} = {val}")
        else:
            for name, val in entry["top_input_features"].items():
                print(f"    {name:45s} = {val}")

        if isinstance(entry["predictions"], dict) and all(isinstance(v, dict) for v in entry["predictions"].values()):
            if labels is not None:
                expected = [k for k, v in entry["predictions"].items() if v.get("actual") == 1]
                predicted = [k for k, v in entry["predictions"].items() if v.get("predicted") == 1]
                print(f"  Expected:  {expected if expected else '[] (healthy)'}")
                print(f"  Predicted: {predicted if predicted else '[] (healthy)'}")
            active = {k: v for k, v in entry["predictions"].items()
                      if v.get("predicted") == 1 or v.get("actual") == 1}
            if active:
                print("  Details:")
                for label, info in active.items():
                    actual_str = f" actual={info['actual']}" if "actual" in info else ""
                    correct_str = f" {'✓' if info.get('correct') else '✗'}" if "correct" in info else ""
                    print(f"    {label:45s} pred={info['predicted']} proba={info.get('probability', '?')}{actual_str}{correct_str}")
        else:
            info = entry["predictions"]
            parts = [f"pred={info['predicted']}"]
            if "probability" in info:
                parts.append(f"proba={info['probability']}")
            if "actual" in info:
                parts.append(f"actual={info['actual']}")
            print(f"  Prediction: {' | '.join(parts)}")
    print(f"{'─'*70}")

    # Counterfactual explanations
    if args.counterfactual and shap_values_per_sample is not None:
        print(f"\n{'═'*70}")
        print("COUNTERFACTUAL ANALYSIS: What changes would flip predictions?")
        print(f"{'═'*70}")
        X = features.reindex(columns=model.feature_names, fill_value=0)
        for i in range(len(features)):
            sv = shap_values_per_sample[i]
            if not hasattr(sv, '__len__') or len(sv) != len(model.feature_names):
                continue
            # Find top features pushing toward failure (positive SHAP for class 1)
            top_idx = np.argsort(sv)[::-1][:5]
            contributors = [(model.feature_names[j], sv[j], float(X.iloc[i, j])) for j in top_idx if sv[j] > 0]
            if contributors:
                print(f"\n  Sample {i} — to flip prediction, reduce these:")
                for fname, shap_val, current_val in contributors:
                    print(f"    {fname:40s} current={current_val:.3f} (SHAP contribution={shap_val:+.4f})")
    elif args.counterfactual:
        print("\n  ⚠️ Counterfactual analysis requires --explain flag (needs SHAP values)")

    print(f"\nLatency: TP50={np.median(latencies_ms):.1f}ms | TP90={np.percentile(latencies_ms, 90):.1f}ms | TP99={np.percentile(latencies_ms, 99):.1f}ms")


if __name__ == "__main__":
    main()
