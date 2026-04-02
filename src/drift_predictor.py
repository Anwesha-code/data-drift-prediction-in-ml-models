"""
drift_predictor.py
------------------
Phase 4 of the project: the drift prediction meta-model.

This reads the saved drift_scores.json and cross_eval_accuracy.json,
joins them into a meta-dataset, trains a regression model to predict
accuracy from divergence scores, and saves the results.

Run from src/:
    python drift_predictor.py
"""

import os
import json
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from config import REPORTS_PATH, MODELS_PATH
from utils import get_logger, save_experiment_report
from datetime import datetime

logger = get_logger("drift_predictor")


def load_meta_dataset():
    """
    Join drift_scores.json and cross_eval_accuracy.json
    into a single meta-dataset for the drift prediction model.
    """
    drift_path = os.path.join(REPORTS_PATH, "drift_scores.json")
    eval_path  = os.path.join(REPORTS_PATH, "cross_eval_accuracy.json")

    with open(drift_path) as f:
        drift_records = json.load(f)

    with open(eval_path) as f:
        eval_records = json.load(f)

    # Build lookup: batch_name -> accuracy
    acc_lookup = {r["batch"]: r["accuracy"] for r in eval_records}

    rows = []
    for rec in drift_records:
        batch = rec["comparison_batch"]
        if batch not in acc_lookup:
            logger.warning(f"No accuracy found for: {batch}")
            continue

        rows.append({
            "batch"              : batch,
            "kl_divergence"      : rec["label_kl_divergence"],
            "js_divergence"      : rec["label_js_divergence"],
            "mean_feature_js"    : rec["mean_feature_js"],
            "accuracy"           : acc_lookup[batch],
        })

    logger.info(f"Meta-dataset built: {len(rows)} samples")
    for r in rows:
        logger.info(
            f"  {r['batch'][:40]:<40} | "
            f"KL={r['kl_divergence']:.4f} | "
            f"JS={r['js_divergence']:.4f} | "
            f"Acc={r['accuracy']:.4f}"
        )
    return rows


def train_drift_predictor(rows: list):
    """
    Train regression models to predict accuracy from drift scores.
    With only 7 samples, we use leave-one-out cross validation
    to get an honest estimate of predictive performance.
    """
    X = np.array([[r["kl_divergence"], r["js_divergence"], r["mean_feature_js"]] for r in rows])
    y = np.array([r["accuracy"] for r in rows])

    feature_names = ["kl_divergence", "js_divergence", "mean_feature_js"]

    models = {
        "linear_regression"       : LinearRegression(),
        "gradient_boosting"       : GradientBoostingRegressor(
                                        n_estimators=50,
                                        max_depth=2,
                                        random_state=42
                                    ),
    }

    results = {}

    for name, model in models.items():
        # Leave-one-out cross validation (necessary with 7 samples)
        loo_preds = []
        loo_true  = []

        for i in range(len(X)):
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i)
            X_test  = X[i:i+1]
            y_test  = y[i]

            model_copy = type(model)(**model.get_params())
            model_copy.fit(X_train, y_train)
            pred = model_copy.predict(X_test)[0]
            loo_preds.append(pred)
            loo_true.append(y_test)

        mae = mean_absolute_error(loo_true, loo_preds)
        r2  = r2_score(loo_true, loo_preds)

        logger.info(f"\n[{name}] LOO-CV MAE={mae:.4f} | R2={r2:.4f}")
        for actual, predicted in zip(loo_true, loo_preds):
            logger.info(f"  actual={actual:.4f}  predicted={predicted:.4f}  error={abs(actual-predicted):.4f}")

        # Train final model on all data
        model.fit(X, y)
        results[name] = {"model": model, "mae": mae, "r2": r2}

        # Save model
        os.makedirs(MODELS_PATH, exist_ok=True)
        model_path = os.path.join(MODELS_PATH, f"drift_predictor_{name}.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Saved -> {model_path}")

        # Save report
        save_experiment_report({
            "Date"          : datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Model"         : f"drift_predictor_{name}",
            "Task"          : "Drift prediction (accuracy regression)",
            "Input features": str(feature_names),
            "Samples"       : len(rows),
            "CV method"     : "Leave-one-out",
            "LOO-CV MAE"    : round(mae, 4),
            "LOO-CV R2"     : round(r2, 4),
        }, filename=f"drift_predictor_{name}.txt")

    return results


def predict_drift_risk(kl: float, js: float, mean_fjs: float,
                       model_name: str = "linear_regression") -> float:
    """
    Given new divergence scores, predict the expected model accuracy.
    Load the saved predictor and return a risk score.
    """
    model_path = os.path.join(MODELS_PATH, f"drift_predictor_{model_name}.pkl")
    model = joblib.load(model_path)
    X_new = np.array([[kl, js, mean_fjs]])
    predicted_accuracy = float(model.predict(X_new)[0])
    predicted_accuracy = max(0.0, min(1.0, predicted_accuracy))
    return predicted_accuracy


if __name__ == "__main__":
    logger.info("=== Drift Predictor Training Started ===")

    rows = load_meta_dataset()

    if len(rows) < 3:
        logger.error("Not enough samples to train. Run testdrift.py first.")
    else:
        results = train_drift_predictor(rows)

        logger.info("\n=== Final Comparison ===")
        for name, res in results.items():
            logger.info(f"{name:<30} MAE={res['mae']:.4f}  R2={res['r2']:.4f}")

        logger.info("\n=== Example Prediction ===")
        # Simulate a new batch with high drift
        acc_pred = predict_drift_risk(kl=0.8, js=0.25, mean_fjs=0.02)
        logger.info(f"For KL=0.8, JS=0.25, mean_fjs=0.02 -> predicted accuracy = {acc_pred:.4f}")

        acc_pred_low = predict_drift_risk(kl=0.01, js=0.004, mean_fjs=0.001)
        logger.info(f"For KL=0.01, JS=0.004, mean_fjs=0.001 -> predicted accuracy = {acc_pred_low:.4f}")

        logger.info("=== Drift Predictor Complete ===")
