"""
drift.py
--------
All drift measurement logic for the project.

Contains:
  - Batch splitting (fixed reference and rolling window)
  - KL Divergence, Jensen-Shannon Divergence, Wasserstein Distance
  - Per-feature drift computation
  - Multi-model cross-dataset evaluation
  - Drift alert system
  - JSON output helpers
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance as scipy_wasserstein
from config import (
    REPORTS_PATH, TARGET_COLUMN,
    COMPUTE_KL, COMPUTE_JS, COMPUTE_WASSERSTEIN,
    DRIFT_ALERT_THRESHOLD, RANDOM_STATE,
)
from utils import get_logger

logger = get_logger("drift")


# ── Batch Splitting ────────────────────────────────────────────────────────────

def split_by_source(df: pd.DataFrame) -> list:
    """
    Split into ordered batches by source file.
    Each batch = one day = one production time window.
    Returns list of (filename, dataframe) tuples.
    """
    from config import RAW_FILES
    batches = []
    for filename in RAW_FILES:
        group = df[df["source_file"] == filename]
        if len(group) == 0:
            logger.warning(f"No rows found for: {filename}")
            continue
        logger.info(f"Batch [{filename}]: {len(group)} rows")
        batches.append((filename, group))
    return batches


# ── Distribution Analysis ──────────────────────────────────────────────────────

def check_distribution(batches: list):
    """Print label distribution for each batch."""
    for name, batch in batches:
        dist = batch[TARGET_COLUMN].value_counts(normalize=True).sort_index()
        logger.info(f"[{name}] Label distribution:\n{dist}\n")


# ── Divergence Metrics ─────────────────────────────────────────────────────────

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL Divergence D_KL(P || Q). Asymmetric."""
    p = np.array(p, dtype=float) + 1e-10
    q = np.array(q, dtype=float) + 1e-10
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon Divergence. Symmetric, bounded [0, 1]."""
    p = np.array(p, dtype=float) + 1e-10
    q = np.array(q, dtype=float) + 1e-10
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def wasserstein_dist(a: np.ndarray, b: np.ndarray) -> float:
    """
    Wasserstein Distance (Earth Mover's Distance) between two 1D arrays.
    No binning required — operates directly on sample values.
    Larger = more distributional shift.
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(scipy_wasserstein(a, b))


# ── Per-Feature Drift ──────────────────────────────────────────────────────────

def compute_feature_drift(batch_a: pd.DataFrame,
                           batch_b: pd.DataFrame) -> dict:
    """
    Compute per-feature JS divergence between two batches.
    Shared bin edges ensure fair comparison across batches.
    Returns {feature_name: js_score}.
    """
    feature_cols = [
        c for c in batch_a.columns
        if c not in [TARGET_COLUMN, "source_file"]
        and pd.api.types.is_numeric_dtype(batch_a[c])
    ]
    drift_scores = {}
    for col in feature_cols:
        combined = pd.concat([batch_a[col], batch_b[col]])
        bins = np.histogram_bin_edges(combined.dropna(), bins=20)
        p, _ = np.histogram(batch_a[col].dropna(), bins=bins, density=True)
        q, _ = np.histogram(batch_b[col].dropna(), bins=bins, density=True)
        drift_scores[col] = round(js_divergence(p, q), 6)
    return drift_scores


def _label_metrics(ref_dist: pd.Series, curr_dist: pd.Series) -> dict:
    """Compute all configured label-level divergence metrics."""
    all_labels = sorted(set(ref_dist.index).union(curr_dist.index))
    ref_vals   = [ref_dist.get(k, 0) for k in all_labels]
    curr_vals  = [curr_dist.get(k, 0) for k in all_labels]
    metrics = {}
    if COMPUTE_KL:
        metrics["label_kl_divergence"] = round(kl_divergence(curr_vals, ref_vals), 6)
    if COMPUTE_JS:
        metrics["label_js_divergence"] = round(js_divergence(curr_vals, ref_vals), 6)
    if COMPUTE_WASSERSTEIN:
        metrics["label_wasserstein"]   = round(wasserstein_dist(curr_vals, ref_vals), 6)
    return metrics


def _feature_wasserstein_mean(ref_batch: pd.DataFrame,
                               curr_batch: pd.DataFrame) -> float:
    """Mean Wasserstein distance across all numeric features."""
    feat_cols = [
        c for c in ref_batch.columns
        if c not in [TARGET_COLUMN, "source_file"]
        and pd.api.types.is_numeric_dtype(ref_batch[c])
    ]
    scores = [
        wasserstein_dist(ref_batch[c].dropna().values,
                         curr_batch[c].dropna().values)
        for c in feat_cols
    ]
    return round(float(np.mean(scores)), 6)


# ── Fixed Reference Drift (Monday vs all) ─────────────────────────────────────

def compute_drift(batches: list) -> list:
    """
    Compare each batch against the fixed reference (Batch 0 = Monday).

    Computes per-comparison:
      - Label-level KL, JS, Wasserstein (as configured)
      - Feature-level mean JS and mean Wasserstein
      - Top 5 most drifted features

    Returns list of result dicts.
    """
    if len(batches) < 2:
        logger.warning("Need at least 2 batches to compute drift.")
        return []

    reference_name, reference_batch = batches[0]
    ref_dist = reference_batch[TARGET_COLUMN].value_counts(
        normalize=True).sort_index()

    results = []

    for i in range(1, len(batches)):
        curr_name, curr_batch = batches[i]
        curr_dist = curr_batch[TARGET_COLUMN].value_counts(
            normalize=True).sort_index()

        label_m     = _label_metrics(ref_dist, curr_dist)
        feature_js  = compute_feature_drift(reference_batch, curr_batch)
        mean_fjs    = round(float(np.mean(list(feature_js.values()))), 6)
        top_drifted = sorted(feature_js, key=feature_js.get, reverse=True)[:5]
        mean_ws     = _feature_wasserstein_mean(reference_batch, curr_batch) \
                      if COMPUTE_WASSERSTEIN else None

        result = {
            "reference_batch"          : reference_name,
            "comparison_batch"         : curr_name,
            "mean_feature_js"          : mean_fjs,
            "mean_feature_wasserstein" : mean_ws,
            "top_drifted_features"     : {f: feature_js[f] for f in top_drifted},
            **label_m,
        }
        results.append(result)

        logger.info(
            f"[Fixed] 0 vs {i} | "
            f"KL={label_m.get('label_kl_divergence', 0):.4f} | "
            f"JS={label_m.get('label_js_divergence', 0):.4f} | "
            f"WS={mean_ws:.4f} | FeatJS={mean_fjs:.4f} | "
            f"{curr_name}"
        )

    return results


# ── Rolling Reference Window ───────────────────────────────────────────────────

def compute_rolling_drift(batches: list) -> list:
    """
    Rolling reference window: compare batch[i] against batch[i-1].

    This simulates a realistic production monitor that checks whether
    the latest data window has shifted relative to the previous window,
    rather than always comparing to the original training distribution.

    Key difference from compute_drift():
      - compute_drift()       : Monday vs Tue, Mon vs Wed, Mon vs Thu ...
      - compute_rolling_drift(): Mon vs Tue, Tue vs Wed, Wed vs Thu ...

    Returns list of result dicts — one per consecutive pair.
    """
    if len(batches) < 2:
        logger.warning("Need at least 2 batches for rolling drift.")
        return []

    logger.info("\n--- Rolling Reference Window Drift ---")
    results = []

    for i in range(1, len(batches)):
        ref_name,  ref_batch  = batches[i - 1]
        curr_name, curr_batch = batches[i]

        ref_dist  = ref_batch[TARGET_COLUMN].value_counts(
            normalize=True).sort_index()
        curr_dist = curr_batch[TARGET_COLUMN].value_counts(
            normalize=True).sort_index()

        label_m     = _label_metrics(ref_dist, curr_dist)
        feature_js  = compute_feature_drift(ref_batch, curr_batch)
        mean_fjs    = round(float(np.mean(list(feature_js.values()))), 6)
        top_drifted = sorted(feature_js, key=feature_js.get, reverse=True)[:5]
        mean_ws     = _feature_wasserstein_mean(ref_batch, curr_batch) \
                      if COMPUTE_WASSERSTEIN else None

        result = {
            "reference_batch"          : ref_name,
            "comparison_batch"         : curr_name,
            "window_type"              : "rolling",
            "mean_feature_js"          : mean_fjs,
            "mean_feature_wasserstein" : mean_ws,
            "top_drifted_features"     : {f: feature_js[f] for f in top_drifted},
            **label_m,
        }
        results.append(result)

        logger.info(
            f"[Rolling] {i-1} -> {i} | "
            f"KL={label_m.get('label_kl_divergence', 0):.4f} | "
            f"JS={label_m.get('label_js_divergence', 0):.4f} | "
            f"WS={mean_ws:.4f} | "
            f"{ref_name} -> {curr_name}"
        )

    return results


# ── Save Results ───────────────────────────────────────────────────────────────

def save_drift_results(results: list,
                        filename: str = "drift_scores.json"):
    os.makedirs(REPORTS_PATH, exist_ok=True)
    path = os.path.join(REPORTS_PATH, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Drift results saved -> {path}")


def save_rolling_drift_results(results: list,
                                filename: str = "rolling_drift_scores.json"):
    os.makedirs(REPORTS_PATH, exist_ok=True)
    path = os.path.join(REPORTS_PATH, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Rolling drift results saved -> {path}")


def save_cross_eval_results(results: list,
                             filename: str = "cross_eval_accuracy.json"):
    os.makedirs(REPORTS_PATH, exist_ok=True)
    path = os.path.join(REPORTS_PATH, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Cross-eval results saved -> {path}")


# ── Multi-Model Cross-Dataset Evaluation ──────────────────────────────────────

def cross_dataset_evaluation(batches: list,
                              model_types: list = None) -> list:
    """
    CORE RESEARCH EXPERIMENT — multi-model version.

    For each model in model_types:
      1. Train on Batch 0 (Monday reference)
      2. Evaluate on every other batch without retraining
      3. Record accuracy per batch

    Scaling is applied internally using the reference batch only,
    to avoid data leakage from test batches into the scaler.
    SVM and Logistic Regression use scaled features.
    Random Forest, XGBoost, Decision Tree do not.

    Returns list of dicts: {model, batch, batch_idx, accuracy, note}
    """
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    from models import get_model

    if model_types is None:
        from config import CROSS_EVAL_MODELS
        model_types = CROSS_EVAL_MODELS

    if len(batches) < 2:
        logger.warning("Need at least 2 batches for cross-dataset evaluation.")
        return []

    reference_name, reference_batch = batches[0]
    X_ref = reference_batch.drop(
        columns=[TARGET_COLUMN, "source_file"], errors="ignore")
    y_ref = reference_batch[TARGET_COLUMN]

    # Scaler fitted ONLY on reference batch
    scaler = StandardScaler()
    X_ref_scaled = scaler.fit_transform(X_ref)

    all_results = []
    NEEDS_SCALING = {"svm", "logistic"}

    # ── Single-class guard ────────────────────────────────────────────────────
    # The Monday (reference) batch may be 100 % class 0 when DEBUG=True
    # loads only a 10 % sample.  LogisticRegression and LinearSVC require
    # at least 2 classes in y_train and raise ValueError if given only one.
    # In that case we fall back to a DummyClassifier that always predicts the
    # majority class — which is exactly what those models would do anyway —
    # and record a note so the behaviour is transparent in the output.
    n_ref_classes = y_ref.nunique()
    if n_ref_classes < 2:
        logger.warning(
            f"Reference batch '{reference_name}' contains only "
            f"{n_ref_classes} class(es) — solvers that require 2+ classes "
            f"(logistic, svm) will use a DummyClassifier fallback."
        )

    from sklearn.dummy import DummyClassifier

    def _fit_model(model_type: str, model, X_train_arr, y_train):
        """
        Fit model or fall back to DummyClassifier when the training
        labels contain fewer classes than the solver requires.

        Returns (fitted_model, was_fallback: bool).
        """
        MULTI_CLASS_REQUIRED = {"logistic", "svm"}
        if model_type in MULTI_CLASS_REQUIRED and y_train.nunique() < 2:
            dummy = DummyClassifier(strategy="most_frequent",
                                    random_state=RANDOM_STATE)
            dummy.fit(X_train_arr, y_train)
            return dummy, True
        model.fit(X_train_arr, y_train)
        return model, False

    for model_type in model_types:
        logger.info(f"\n--- Cross-eval model: {model_type} ---")
        model = get_model(model_type)

        X_train = X_ref_scaled if model_type in NEEDS_SCALING else X_ref.values
        fitted_model, used_fallback = _fit_model(model_type, model, X_train, y_ref)

        if used_fallback:
            logger.warning(
                f"  [{model_type}] Reference is single-class — "
                f"using DummyClassifier (always predicts majority class)."
            )

        # Self-accuracy (upper bound)
        self_acc = accuracy_score(y_ref, fitted_model.predict(X_train))
        self_note = (
            "same distribution (upper bound) [DummyClassifier — single-class ref]"
            if used_fallback else
            "same distribution (upper bound)"
        )
        all_results.append({
            "model"    : model_type,
            "batch"    : reference_name,
            "batch_idx": 0,
            "accuracy" : round(self_acc, 4),
            "note"     : self_note,
        })
        logger.info(f"  Self-accuracy: {self_acc:.4f}")

        for i in range(1, len(batches)):
            curr_name, curr_batch = batches[i]
            X_test_raw = curr_batch.drop(
                columns=[TARGET_COLUMN, "source_file"], errors="ignore")
            y_test = curr_batch[TARGET_COLUMN]

            shared = [c for c in X_ref.columns if c in X_test_raw.columns]
            X_test_raw = X_test_raw[shared]

            X_test = scaler.transform(X_test_raw) \
                     if model_type in NEEDS_SCALING else X_test_raw.values

            acc = accuracy_score(y_test, fitted_model.predict(X_test))
            all_results.append({
                "model"    : model_type,
                "batch"    : curr_name,
                "batch_idx": i,
                "accuracy" : round(acc, 4),
                "note"     : (
                    "cross-dataset (drift scenario) [DummyClassifier]"
                    if used_fallback else
                    "cross-dataset (drift scenario)"
                ),
            })
            logger.info(f"  Batch {i} [{curr_name[:35]}]: {acc:.4f}")

    return all_results


# ── Drift Alert System ─────────────────────────────────────────────────────────

def check_drift_alerts(cross_eval_results: list,
                        drift_scores: list) -> list:
    """
    Check each batch's average cross-eval accuracy against
    DRIFT_ALERT_THRESHOLD. Writes drift_alerts.json if any
    batches fall below the threshold.

    Also calls the saved drift predictor if available.
    Returns list of alert dicts.
    """
    import joblib
    from config import MODELS_PATH

    drift_lookup = {r["comparison_batch"]: r for r in drift_scores}

    # Average accuracy per batch across all models
    batch_accs: dict = {}
    batch_per_model: dict = {}
    for r in cross_eval_results:
        b = r["batch"]
        if b not in batch_accs:
            batch_accs[b]     = []
            batch_per_model[b] = {}
        batch_accs[b].append(r["accuracy"])
        batch_per_model[b][r["model"]] = r["accuracy"]

    predictor = None
    predictor_path = os.path.join(MODELS_PATH,
                                   "drift_predictor_linear_regression.pkl")
    if os.path.exists(predictor_path):
        try:
            predictor = joblib.load(predictor_path)
        except Exception as e:
            logger.warning(f"Could not load drift predictor: {e}")

    alerts = []
    for batch, accs in batch_accs.items():
        avg_acc = float(np.mean(accs))
        if avg_acc >= DRIFT_ALERT_THRESHOLD:
            continue

        drift_info   = drift_lookup.get(batch, {})
        kl           = drift_info.get("label_kl_divergence")
        js           = drift_info.get("label_js_divergence")
        mfjs         = drift_info.get("mean_feature_js")
        predicted    = None

        if predictor and kl is not None and js is not None and mfjs is not None:
            try:
                predicted = round(float(
                    np.clip(predictor.predict([[kl, js, mfjs]])[0], 0, 1)
                ), 4)
            except Exception:
                pass

        alert = {
            "batch"              : batch,
            "average_accuracy"   : round(avg_acc, 4),
            "threshold"          : DRIFT_ALERT_THRESHOLD,
            "per_model_accuracy" : batch_per_model[batch],
            "label_kl_divergence": kl,
            "label_js_divergence": js,
            "predicted_accuracy" : predicted,
            "alert"              : "DRIFT DETECTED — avg accuracy below threshold",
        }
        alerts.append(alert)
        logger.warning(
            f"DRIFT ALERT | {batch[:40]} | "
            f"avg_acc={avg_acc:.4f} < {DRIFT_ALERT_THRESHOLD}"
        )

    if alerts:
        path = os.path.join(REPORTS_PATH, "drift_alerts.json")
        os.makedirs(REPORTS_PATH, exist_ok=True)
        with open(path, "w") as f:
            json.dump(alerts, f, indent=4)
        logger.info(f"Alerts saved -> {path}")
    else:
        logger.info("No drift alerts triggered.")

    return alerts
