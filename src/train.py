"""
train.py
--------
Baseline model training pipeline.

Trains all models in MODELS_TO_RUN (config.py) on the full combined
processed dataset, evaluates on the held-out test split, saves each
model and its experiment report.

Decision Tree and SVM are now included alongside the original three.

Note on SVM training time:
  LinearSVC on 2.26M training samples takes approximately 5-15 minutes
  depending on hardware. It is placed last in MODELS_TO_RUN so the
  other models complete quickly before it runs.

Run from src/:
  python main.py
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from datetime import datetime

from config import *
from preprocessing import run_preprocessing, scale_features
from models import get_model
from utils import get_logger, save_experiment_report

logger = get_logger("train")


def train_pipeline():
    logger.info("=== Training Pipeline Started ===")

    # ── Load or generate processed data ───────────────────────────────────
    processed_path = os.path.join(PROCESSED_DATA_PATH, PROCESSED_FILENAME)

    if os.path.exists(processed_path):
        logger.info(f"Loading existing processed data: {processed_path}")
        df = pd.read_csv(processed_path)
    else:
        logger.info("Processed data not found. Running preprocessing...")
        df = run_preprocessing()

    # ── Split features / target ────────────────────────────────────────────
    X = df.drop(columns=[TARGET_COLUMN, "source_file"], errors="ignore")
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)} | Features: {X.shape[1]}")

    # ── Scale features ─────────────────────────────────────────────────────
    # Scaler is always fitted so that SVM and Logistic get proper input.
    # Tree models receive the scaled arrays too — scaling does not affect
    # their predictions, but it keeps the pipeline uniform.
    X_train_sc, X_test_sc, _ = scale_features(X_train, X_test)

    # Models that require scaled input
    NEEDS_SCALING = {"svm", "logistic"}

    # ── Train each model ───────────────────────────────────────────────────
    for model_type in MODELS_TO_RUN:
        logger.info(f"\n=== Training: {model_type} ===")

        model   = get_model(model_type)
        X_tr    = X_train_sc if model_type in NEEDS_SCALING else X_train.values
        X_te    = X_test_sc  if model_type in NEEDS_SCALING else X_test.values

        model.fit(X_tr, y_train)

        preds   = model.predict(X_te)
        acc     = accuracy_score(y_test, preds)
        report  = classification_report(y_test, preds)
        cm      = confusion_matrix(y_test, preds)

        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"\n{report}")
        logger.info(f"Confusion Matrix:\n{cm}")

        # Save predictions CSV
        os.makedirs(REPORTS_PATH, exist_ok=True)
        pred_df = pd.DataFrame({"y_true": y_test, "y_pred": preds})
        pred_df.to_csv(
            os.path.join(REPORTS_PATH, f"predictions_{model_type}.csv"),
            index=False
        )

        # Save model artifact
        if SAVE_MODEL:
            os.makedirs(MODELS_PATH, exist_ok=True)
            model_path = os.path.join(MODELS_PATH, f"{model_type}_v1.pkl")
            joblib.dump(model, model_path)
            logger.info(f"Model saved -> {model_path}")

        # Save experiment report
        save_experiment_report({
            "Date"          : datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Model"         : model_type,
            "Dataset"       : PROCESSED_FILENAME,
            "Features"      : X.shape[1],
            "Train Samples" : len(X_train),
            "Test Samples"  : len(X_test),
            "Accuracy"      : round(acc, 4),
            "Scaled"        : model_type in NEEDS_SCALING,
        })


if __name__ == "__main__":
    train_pipeline()
