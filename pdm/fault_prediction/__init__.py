"""Fault prediction submodule (AutoGluon supervised models)."""

import pandas as pd

from pdm.data.utils import encode_categoricals, booleans_to_int


def baseline_engineer_features(
    raw_df: pd.DataFrame, drop_cols: list[str] | None = None
) -> pd.DataFrame:
    """Minimal baseline feature engineering: no domain knowledge applied.

    Pure, deterministic transforms only — no data-dependent feature selection.
    Produces the same columns regardless of input size.

    Args:
        raw_df: Raw dataset (may include IDs, dates, labels).
        drop_cols: Columns to drop (e.g. ["device_id", "_observation_date"]).
    """
    df = raw_df.copy()
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    label_cols = [c for c in df.columns if c.startswith("label_")]
    feature_cols = [c for c in df.columns if c not in label_cols]

    cat_cols = [c for c in feature_cols if df[c].dtype == "object"]
    if cat_cols:
        df = encode_categoricals(df, cat_cols)

    # Recompute after encoding (original cat cols replaced with *_enc)
    feature_cols = [c for c in df.columns if not c.startswith("label_")]
    bool_cols = [c for c in feature_cols if df[c].dtype == "bool"]
    if bool_cols:
        df = booleans_to_int(df, bool_cols)

    df = df.fillna(0)
    return df
