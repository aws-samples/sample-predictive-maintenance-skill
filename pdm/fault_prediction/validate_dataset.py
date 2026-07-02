# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas>=2.0", "numpy>=1.24"]
# ///
"""Validate dataset.csv before training. Run as a gate between Phase 3 and Phase 4.

Supports:
- Binary classification (single target column)
- Multi-label classification (multiple label_ columns)
- RUL regression (unit_id + RUL columns)
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path


def validate(path: str, target_col: str | None = None) -> bool:
    """Return True if dataset passes all checks, False otherwise."""
    df = pd.read_csv(path)
    errors = []
    warnings = []
    checks_passed = []

    # --- Auto-detect formulation ---
    label_cols = [c for c in df.columns if c.startswith("label_")]

    if label_cols:
        # Multi-label mode
        return _validate_multilabel(df, label_cols, errors, warnings, checks_passed)

    # Single-target mode
    if target_col is None:
        candidates = [c for c in df.columns if c.lower() in ("rul", "machine_failure", "failure", "target", "label")]
        if not candidates:
            errors.append("Cannot auto-detect target column. Pass --target explicitly.")
            _report(errors, warnings, df, checks_passed)
            return False
        target_col = candidates[0]

    if target_col not in df.columns:
        errors.append(f"Target column '{target_col}' not found. Columns: {list(df.columns[:10])}")
        _report(errors, warnings, df, checks_passed)
        return False

    # Check 1: No NaN in target
    nan_target = df[target_col].isna().sum()
    if nan_target > 0:
        errors.append(f"Target '{target_col}' has {nan_target} NaN values ({nan_target/len(df)*100:.1f}%)")
    else:
        checks_passed.append(f"No NaN in target column '{target_col}'")

    # Check 2: All columns numeric
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    non_numeric = [c for c in non_numeric if c not in ("unit_id",)]  # unit_id allowed for RUL
    if non_numeric:
        errors.append(f"Non-numeric columns (drop or encode): {non_numeric}")
    else:
        checks_passed.append(f"All {df.shape[1]} columns are numeric")

    # Check 3: No constant columns (zero-variance)
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    zero_var = [c for c in feature_cols if df[c].nunique() <= 1]
    if zero_var:
        warnings.append(f"Zero-variance features ({len(zero_var)}): {zero_var[:5]}")
    else:
        checks_passed.append(f"No zero-variance features (checked {len(feature_cols)} columns)")

    # Check 4: ID leakage — features that correlate >0.95 with row index
    idx = np.arange(len(df))
    leaky = []
    for c in feature_cols[:50]:  # check first 50 for speed
        if df[c].nunique() > 10:
            corr = np.corrcoef(idx, df[c].fillna(0).values)[0, 1]
            if abs(corr) > 0.95:
                leaky.append(c)
    if leaky:
        warnings.append(f"Possible ID/index leakage (corr>0.95 with row order): {leaky}")
    else:
        checks_passed.append("No ID/index leakage detected")

    # Check 5: Class balance (classification) or RUL range (regression)
    is_rul = target_col.lower() == "rul" or df[target_col].nunique() > 20
    if is_rul:
        rul_range = df[target_col].max() - df[target_col].min()
        print(f"  RUL range: {df[target_col].min():.0f} – {df[target_col].max():.0f} (range={rul_range:.0f})")
        if df[target_col].min() < 0:
            errors.append(f"Negative RUL values found (min={df[target_col].min():.1f})")
        else:
            checks_passed.append(f"RUL range valid: {df[target_col].min():.0f}–{df[target_col].max():.0f}")
    else:
        pos_rate = df[target_col].mean()
        print(f"  Positive rate: {pos_rate:.1%} ({int(df[target_col].sum())}/{len(df)})")
        if pos_rate < 0.02:
            warnings.append(f"Extreme class imbalance: {pos_rate:.2%} positive. Consider class weighting.")
        elif pos_rate < 0.05:
            warnings.append(f"Class imbalance: {pos_rate:.1%} positive. class_weight='balanced' recommended.")
        else:
            checks_passed.append(f"Class balance OK: {pos_rate:.1%} positive")

    # Check 6: High NaN features
    nan_pct = df[feature_cols].isna().mean()
    high_nan = nan_pct[nan_pct > 0.5]
    if len(high_nan) > 0:
        warnings.append(f"Features with >50% NaN ({len(high_nan)}): {list(high_nan.index[:5])}")
    else:
        checks_passed.append("No features with >50% NaN")

    _report(errors, warnings, df, checks_passed)
    return len(errors) == 0


def _validate_multilabel(df: pd.DataFrame, label_cols: list[str],
                         errors: list, warnings: list, checks_passed: list) -> bool:
    """Validate a multi-label dataset (columns prefixed with label_)."""
    print(f"  Detected multi-label dataset: {len(label_cols)} labels")
    feature_cols = [c for c in df.columns if c not in label_cols]

    # Check 1: No NaN in any label column
    label_nans = df[label_cols].isna().sum()
    nan_labels = label_nans[label_nans > 0]
    if len(nan_labels) > 0:
        errors.append(f"Labels with NaN: {dict(nan_labels)}")
    else:
        checks_passed.append(f"No NaN in any of {len(label_cols)} label columns")

    # Check 2: Labels are binary (0 or 1)
    non_binary = []
    for col in label_cols:
        unique_vals = set(df[col].dropna().unique())
        if not unique_vals.issubset({0, 1, 0.0, 1.0}):
            non_binary.append(col)
    if non_binary:
        errors.append(f"Non-binary label columns: {non_binary[:5]}")
    else:
        checks_passed.append("All label columns are binary (0/1)")

    # Check 3: All feature columns numeric
    non_numeric = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        errors.append(f"Non-numeric feature columns (drop or encode): {non_numeric}")
    else:
        checks_passed.append(f"All {len(feature_cols)} feature columns are numeric")

    # Check 4: Zero-variance features
    numeric_features = [c for c in feature_cols if df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
    zero_var = [c for c in numeric_features if df[c].nunique() <= 1]
    if zero_var:
        warnings.append(f"Zero-variance features ({len(zero_var)}): {zero_var[:5]}")
    else:
        checks_passed.append(f"No zero-variance features (checked {len(numeric_features)} columns)")

    # Check 5: ID leakage
    idx = np.arange(len(df))
    leaky = []
    for c in numeric_features[:50]:
        if df[c].nunique() > 10:
            corr = np.corrcoef(idx, df[c].fillna(0).values)[0, 1]
            if abs(corr) > 0.95:
                leaky.append(c)
    if leaky:
        warnings.append(f"Possible ID/index leakage (corr>0.95 with row order): {leaky}")
    else:
        checks_passed.append("No ID/index leakage detected")

    # Check 6: Per-label positive rates
    print(f"\n  Per-label positive rates:")
    low_pos_labels = []
    for col in label_cols:
        pos_rate = df[col].mean()
        n_pos = int(df[col].sum())
        status = "⚠️" if pos_rate < 0.005 else "  "
        print(f"  {status} {col}: {pos_rate:.3%} ({n_pos} positives)")
        if pos_rate < 0.005:
            low_pos_labels.append(col)
    if low_pos_labels:
        warnings.append(f"Very low positive rate (<0.5%) in {len(low_pos_labels)} labels: {low_pos_labels[:5]}. "
                        "Consider dropping or grouping these.")

    # Check 7: High NaN features
    nan_pct = df[numeric_features].isna().mean()
    high_nan = nan_pct[nan_pct > 0.5]
    if len(high_nan) > 0:
        warnings.append(f"Features with >50% NaN ({len(high_nan)}): {list(high_nan.index[:5])}")
    else:
        checks_passed.append("No features with >50% NaN")

    _report(errors, warnings, df, checks_passed)
    return len(errors) == 0


def _report(errors: list, warnings: list, df: pd.DataFrame, checks_passed: list | None = None):
    print(f"\n{'='*60}")
    print(f"Dataset Validation Report")
    print(f"{'='*60}")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    if checks_passed:
        print(f"\n  Checks run:")
        for check in checks_passed:
            print(f"   ✅ {check}")
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}) — must fix before training:")
        for e in errors:
            print(f"   • {e}")
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"   • {w}")
    if not errors and not warnings:
        print("\n✅ All checks passed.")
    elif not errors:
        print("\n✅ No blocking errors. Proceed with training.")
    else:
        print("\n🛑 Fix errors above before running train.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate PdM dataset before training")
    parser.add_argument("--data", default="./data/dataset.csv", help="Path to dataset CSV")
    parser.add_argument("--target", default=None, help="Target column name (auto-detected if omitted)")
    parser.add_argument("--fix", action="store_true", help="Auto-fix common issues (drop zero-variance, encode categoricals, fill NaN)")
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"❌ File not found: {args.data}")
        sys.exit(1)

    ok = validate(args.data, args.target)

    if not ok and args.fix:
        print("\n🔧 Attempting auto-fix...")
        df = pd.read_csv(args.data)
        n_before = df.shape[1]

        # Drop zero-variance
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        label_cols = [c for c in df.columns if c.startswith("label_")]
        protected = set(label_cols) | {"RUL", "machine_failure", "duration", "event", "unit_id"}
        droppable = [c for c in numeric_cols if c not in protected and df[c].nunique() <= 1]
        if droppable:
            df = df.drop(columns=droppable)
            print(f"  Dropped {len(droppable)} zero-variance columns")

        # Encode remaining non-numeric (except unit_id)
        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
        non_numeric = [c for c in non_numeric if c != "unit_id"]
        for col in non_numeric:
            df[col] = pd.factorize(df[col])[0]
        if non_numeric:
            print(f"  Label-encoded {len(non_numeric)} non-numeric columns: {non_numeric[:5]}")

        # Fill NaN
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            df = df.fillna(0)
            print(f"  Filled {nan_count} NaN values with 0")

        df.to_csv(args.data, index=False)
        print(f"  Saved fixed dataset: {df.shape[0]} rows × {df.shape[1]} cols (was {n_before} cols)")
        print("\n  Re-validating...")
        ok = validate(args.data, args.target)

    sys.exit(0 if ok else 1)
