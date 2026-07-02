"""Auto-generate interaction features from numeric column pairs.

Produces pairwise products, ratios, and differences — the type of features
that gave +9.3% F1 on AI4I when hand-crafted (power, overstrain, temp_diff).

Usage:
    from pdm.data.interactions import generate_interactions

    train_aug, new_cols = generate_interactions(
        train_df, feature_cols, target_col="machine_failure", max_features=30,
    )
"""
import itertools
from typing import Optional

import numpy as np
import pandas as pd


def generate_interactions(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: Optional[str] = None,
    operators: list[str] | None = None,
    max_features: int = 50,
    min_correlation: float = 0.05,
    max_input_features: int = 15,
) -> tuple[pd.DataFrame, list[str]]:
    """Generate pairwise interaction features and select the most useful.

    Args:
        df: Input DataFrame with features and optional target
        feature_cols: Numeric columns to combine pairwise
        target_col: If provided, filter interactions by correlation with target
        operators: Which operations to apply. Default: ["multiply", "ratio", "subtract"]
        max_features: Maximum number of interaction features to keep
        min_correlation: Minimum |correlation| with target to retain a feature
        max_input_features: Pre-filter to top N features (by target correlation) before generating pairs

    Returns:
        (augmented_df, new_feature_names): DataFrame with original + new columns, and list of new column names
    """
    if operators is None:
        operators = ["multiply", "ratio", "subtract"]

    # Only use numeric columns
    numeric_cols = [c for c in feature_cols if df[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)]

    # Pre-filter to most relevant features if too many
    if target_col and target_col in df.columns and len(numeric_cols) > max_input_features:
        correlations = df[numeric_cols].corrwith(df[target_col]).abs()
        numeric_cols = correlations.nlargest(max_input_features).index.tolist()

    # Generate interactions
    new_features = {}
    for col_a, col_b in itertools.combinations(numeric_cols, 2):
        a = df[col_a].values.astype(float)
        b = df[col_b].values.astype(float)

        if "multiply" in operators:
            new_features[f"{col_a}_x_{col_b}"] = a * b

        if "ratio" in operators:
            # Safe ratio: avoid division by zero
            denom = np.where(np.abs(b) > 1e-10, b, 1e-10)
            new_features[f"{col_a}_div_{col_b}"] = a / denom

        if "subtract" in operators:
            new_features[f"{col_a}_minus_{col_b}"] = a - b

    if not new_features:
        return df.copy(), []

    interactions_df = pd.DataFrame(new_features, index=df.index)

    # Remove constant columns
    interactions_df = interactions_df.loc[:, interactions_df.std() > 1e-10]

    # Filter by correlation with target
    if target_col and target_col in df.columns and len(interactions_df.columns) > 0:
        target_values = df[target_col]
        correlations = interactions_df.corrwith(target_values).abs()
        # Keep only features above minimum correlation
        keep = correlations[correlations >= min_correlation].nlargest(max_features).index.tolist()
        interactions_df = interactions_df[keep]

    # Final selection: top max_features by variance (if no target)
    if (target_col is None or target_col not in df.columns) and len(interactions_df.columns) > max_features:
        top_var = interactions_df.var().nlargest(max_features).index.tolist()
        interactions_df = interactions_df[top_var]

    new_names = list(interactions_df.columns)
    result = pd.concat([df, interactions_df], axis=1)
    return result, new_names
