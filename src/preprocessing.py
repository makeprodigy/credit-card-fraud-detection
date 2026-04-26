"""
preprocessing.py
----------------
Preprocessing pipeline for the Credit Card Fraud Detection project.

The dataset contains:
  - Time     : seconds elapsed (needs scaling)
  - V1–V28   : PCA-transformed features (already scaled by PCA)
  - Amount   : transaction amount (needs scaling)
  - Class    : 0 = legitimate, 1 = fraud  (target)

Strategy
--------
1. Scale `Time` and `Amount` with RobustScaler (robust to outliers).
2. Leave V1–V28 as-is (PCA already zero-centres and unit-normalises them).
3. Combine with ColumnTransformer so the full feature matrix is returned.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

# Features that need scaling
SCALE_COLS = ["Time", "Amount"]

# PCA features — pass through untouched
V_COLS = [f"V{i}" for i in range(1, 29)]

# All feature columns in order
FEATURE_COLS = ["Time"] + V_COLS + ["Amount"]


def build_preprocessor() -> ColumnTransformer:
    """
    Returns a fitted-ready ColumnTransformer that:
      - RobustScales Time and Amount
      - Passes V1–V28 through unchanged
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("scale", RobustScaler(), SCALE_COLS),
            ("passthrough", "passthrough", V_COLS),
        ],
        remainder="drop",        # drop any unexpected columns
        verbose_feature_names_out=False,
    )
    return preprocessor


def load_data(filepath: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load CSV and return (X, y)."""
    df = pd.read_csv(filepath)
    X = df[FEATURE_COLS]
    y = df["Class"]
    return X, y
