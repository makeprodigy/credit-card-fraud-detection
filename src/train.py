"""
train.py
--------
End-to-end training script for the Credit Card Fraud Detection project.

Models trained:
  1. Logistic Regression  (linear baseline)
  2. Decision Tree        (explainable rules)
  3. Random Forest        (ensemble, best performance)

Each model is wrapped in an sklearn Pipeline:
  preprocessor  →  SMOTE (training only)  →  classifier

Artifacts saved to models/:
  logistic_regression.joblib
  decision_tree.joblib
  random_forest.joblib

Usage
-----
  python src/train.py                    # uses default data path
  python src/train.py --data creditcard.csv
"""

import argparse
import os
import sys
import time

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Ensure src/ is on the path when running from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import build_preprocessor, load_data, FEATURE_COLS
from src.evaluate import evaluate_model, compare_models

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_PATH = "creditcard.csv"
MODELS_DIR = "models"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def build_pipeline(classifier, use_smote: bool = True):
    """
    Build an imbalanced-learn Pipeline:
      1. Preprocessing (RobustScaler on Time/Amount, passthrough V1-V28)
      2. SMOTE (optional — skip for tree-based if you prefer class_weight)
      3. Classifier
    """
    steps = [("preprocessor", build_preprocessor())]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.1)))
    steps.append(("classifier", classifier))
    return ImbPipeline(steps=steps)


def get_models() -> dict:
    """Return a dict of {name: pipeline} for all models."""
    return {
        "logistic_regression": build_pipeline(
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                solver="lbfgs",
            ),
            use_smote=True,
        ),
        "decision_tree": build_pipeline(
            DecisionTreeClassifier(
                max_depth=8,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                min_samples_leaf=10,
            ),
            use_smote=True,
        ),
        "random_forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                min_samples_leaf=5,
            ),
            use_smote=False,   # RF handles imbalance well with class_weight
        ),
    }


def main(data_path: str = DATA_PATH):
    print(f"\n{'▓'*55}")
    print("  Credit Card Fraud Detection — Training Pipeline")
    print(f"{'▓'*55}\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"[1/4] Loading data from: {data_path}")
    X, y = load_data(data_path)
    print(f"      Rows: {len(X):,}  |  Features: {X.shape[1]}  |  Fraud rate: {y.mean():.4%}")

    # ── Train/test split ───────────────────────────────────────────────────────
    print(f"\n[2/4] Splitting data  (test={TEST_SIZE*100:.0f}%, stratified)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── Train & evaluate ───────────────────────────────────────────────────────
    print(f"\n[3/4] Training models …")
    os.makedirs(MODELS_DIR, exist_ok=True)

    models = get_models()
    results = []

    for name, pipeline in models.items():
        print(f"\n  ⟳  {name.replace('_', ' ').title()} …", flush=True)
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"     Trained in {elapsed:.1f}s")

        metrics = evaluate_model(pipeline, X_test, y_test, model_name=name.replace("_", " ").title())
        results.append(metrics)

        # Save model artifact
        model_path = os.path.join(MODELS_DIR, f"{name}.joblib")
        joblib.dump(pipeline, model_path)
        print(f"     Saved → {model_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n[4/4] Model Comparison (sorted by AUPRC)")
    comparison = compare_models(results)
    print(comparison.to_string())

    print(f"\n✓ All models saved to ./{MODELS_DIR}/")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument("--data", default=DATA_PATH, help="Path to creditcard.csv")
    args = parser.parse_args()
    main(data_path=args.data)
