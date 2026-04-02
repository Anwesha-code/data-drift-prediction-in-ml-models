import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime

from config import *
from preprocessing import run_preprocessing, scale_features
from models import get_model
from utils import get_logger, save_experiment_report

logger = get_logger("train")


def train_pipeline():
    logger.info("=== Training Pipeline Started ===")

    processed_path = os.path.join(PROCESSED_DATA_PATH, PROCESSED_FILENAME)

    if os.path.exists(processed_path):
        logger.info(f"Loading existing processed data from: {processed_path}")
        df = pd.read_csv(processed_path)
    else:
        logger.info("Processed data not found. Running preprocessing...")
        df = run_preprocessing()

    X = df.drop(columns=[TARGET_COLUMN, "source_file"], errors="ignore")
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    logger.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    if SCALE_FEATURES:
        X_train, X_test, _ = scale_features(X_train, X_test)

    # MULTI-MODEL LOOP
    for model_type in MODELS_TO_RUN:
        logger.info(f"\n=== Training: {model_type} ===")

        model = get_model(model_type)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"\n{classification_report(y_test, preds)}")

        # Confusion matrix
        cm = confusion_matrix(y_test, preds)
        logger.info(f"Confusion Matrix:\n{cm}")

        # Save predictions
        os.makedirs(REPORTS_PATH, exist_ok=True)
        pred_df = pd.DataFrame({"y_true": y_test, "y_pred": preds})
        pred_path = os.path.join(REPORTS_PATH, f"predictions_{model_type}.csv")
        pred_df.to_csv(pred_path, index=False)

        # Save model
        if SAVE_MODEL:
            os.makedirs(MODELS_PATH, exist_ok=True)
            model_path = os.path.join(MODELS_PATH, f"{model_type}_v1.pkl")
            joblib.dump(model, model_path)
            logger.info(f"Model saved -> {model_path}")

        # Save report
        save_experiment_report({
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Model": model_type,
            "Dataset": PROCESSED_FILENAME,
            "Features": X.shape[1],
            "Train Samples": len(X_train),
            "Test Samples": len(X_test),
            "Accuracy": round(acc, 4),
        })


if __name__ == "__main__":
    train_pipeline()