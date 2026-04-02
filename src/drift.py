# KL divergence

# JS divergence

# drift detection logic
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from config import REPORTS_PATH, TARGET_COLUMN
from utils import get_logger

logger = get_logger("drift")


# ─── Batch Splitting ──────────────────────────────────────────────────────────

def split_by_source(df: pd.DataFrame) -> list:
    """
    Split the combined dataframe into ordered batches by source file.
    Each batch = one day = one production time window.
    Order is preserved based on the RAW_FILES list in config.
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


# ─── Distribution Analysis ────────────────────────────────────────────────────

def check_distribution(batches: list):
    """Print label distribution for each batch."""
    for name, batch in batches:
        dist = batch[TARGET_COLUMN].value_counts(normalize=True).sort_index()
        logger.info(f"[{name}] Label distribution:\n{dist}\n")


# ─── Divergence Metrics ───────────────────────────────────────────────────────

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    KL Divergence: D_KL(P || Q)
    Measures how much distribution P diverges from reference Q.
    Asymmetric — order matters.
    Small epsilon added to avoid log(0).
    """
    p = np.array(p, dtype=float) + 1e-10
    q = np.array(q, dtype=float) + 1e-10
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon Divergence: symmetric version of KL.
    Always between 0 and 1 (when using log base 2).
    More stable and preferred for drift measurement.
    """
    p = np.array(p, dtype=float) + 1e-10
    q = np.array(q, dtype=float) + 1e-10
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def compute_feature_drift(batch_a: pd.DataFrame, batch_b: pd.DataFrame) -> dict:
    """
    Compute per-feature drift between two batches using JS divergence.
    Uses histogram binning to convert continuous features to distributions.
    Returns a dict of {feature_name: js_score}.
    """
    feature_cols = [c for c in batch_a.columns
                    if c not in [TARGET_COLUMN, "source_file"]
                    and pd.api.types.is_numeric_dtype(batch_a[c])]

    drift_scores = {}

    for col in feature_cols:
        # Create shared bin edges for fair comparison
        combined = pd.concat([batch_a[col], batch_b[col]])
        bins = np.histogram_bin_edges(combined.dropna(), bins=20)

        p, _ = np.histogram(batch_a[col].dropna(), bins=bins, density=True)
        q, _ = np.histogram(batch_b[col].dropna(), bins=bins, density=True)

        drift_scores[col] = round(js_divergence(p, q), 6)

    return drift_scores


# ─── Main Drift Computation ───────────────────────────────────────────────────

def compute_drift(batches: list) -> list:
    """
    Compare each batch against the reference (Batch 0 = Monday).
    Computes:
      - Label distribution KL and JS divergence
      - Per-feature JS divergence (mean across all features)
    Returns a list of result dicts — one per comparison.
    """
    if len(batches) < 2:
        logger.warning("Need at least 2 batches to compute drift.")
        return []

    reference_name, reference_batch = batches[0]
    ref_dist = reference_batch[TARGET_COLUMN].value_counts(normalize=True).sort_index()

    results = []

    for i in range(1, len(batches)):
        curr_name, curr_batch = batches[i]
        curr_dist = curr_batch[TARGET_COLUMN].value_counts(normalize=True).sort_index()

        # Align label distributions
        all_labels = sorted(set(ref_dist.index).union(curr_dist.index))
        ref_vals   = [ref_dist.get(k, 0) for k in all_labels]
        curr_vals  = [curr_dist.get(k, 0) for k in all_labels]

        kl  = kl_divergence(ref_vals, curr_vals)
        js  = js_divergence(ref_vals, curr_vals)

        # Per-feature drift (mean JS across all features)
        feature_drift = compute_feature_drift(reference_batch, curr_batch)
        mean_feature_js = float(np.mean(list(feature_drift.values())))
        top_drifted = sorted(feature_drift, key=feature_drift.get, reverse=True)[:5]

        result = {
            "reference_batch"   : reference_name,
            "comparison_batch"  : curr_name,
            "label_kl_divergence" : round(kl, 6),
            "label_js_divergence" : round(js, 6),
            "mean_feature_js"   : round(mean_feature_js, 6),
            "top_drifted_features": {f: feature_drift[f] for f in top_drifted},
        }

        results.append(result)

        logger.info(
            f"Drift [Batch 0 vs Batch {i}] | "
            f"KL={kl:.4f} | JS={js:.4f} | "
            f"Mean Feature JS={mean_feature_js:.4f} | "
            f"Comparing: {reference_name} vs {curr_name}"
        )

    return results


# ─── Save Drift Results ───────────────────────────────────────────────────────

def save_drift_results(results: list, filename: str = "drift_scores.json"):
    """Save drift computation results to artifacts/reports/."""
    os.makedirs(REPORTS_PATH, exist_ok=True)
    path = os.path.join(REPORTS_PATH, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Drift results saved -> {path}")


# ─── Cross-Dataset Evaluation ─────────────────────────────────────────────────

def cross_dataset_evaluation(batches: list) -> list:
    """
    CORE RESEARCH EXPERIMENT:
    Train on Batch 0 (reference), evaluate on every other batch.
    This directly shows how model accuracy drops as drift increases.
    Returns list of {batch, accuracy} records.
    """
    from sklearn.metrics import accuracy_score
    from models import get_model

    if len(batches) < 2:
        logger.warning("Need at least 2 batches for cross-dataset evaluation.")
        return []

    reference_name, reference_batch = batches[0]

    X_train = reference_batch.drop(columns=[TARGET_COLUMN, "source_file"], errors="ignore")
    y_train = reference_batch[TARGET_COLUMN]

    model = get_model("random_forest")
    logger.info(f"Training on reference batch: {reference_name}")
    model.fit(X_train, y_train)

    eval_results = []

    # Evaluate on reference itself (upper bound)
    preds_ref = model.predict(X_train)
    acc_ref   = accuracy_score(y_train, preds_ref)
    eval_results.append({
        "batch"    : reference_name,
        "batch_idx": 0,
        "accuracy" : round(acc_ref, 4),
        "note"     : "same distribution (upper bound)"
    })
    logger.info(f"Self-accuracy on reference: {acc_ref:.4f}")

    # Evaluate on all other batches
    for i in range(1, len(batches)):
        curr_name, curr_batch = batches[i]

        X_test = curr_batch.drop(columns=[TARGET_COLUMN, "source_file"], errors="ignore")
        y_test = curr_batch[TARGET_COLUMN]

        # Align columns — in case feature engineering dropped different cols
        shared_cols = [c for c in X_train.columns if c in X_test.columns]
        X_test = X_test[shared_cols]

        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)

        eval_results.append({
            "batch"    : curr_name,
            "batch_idx": i,
            "accuracy" : round(acc, 4),
            "note"     : "cross-dataset (drift scenario)"
        })

        logger.info(f"Cross-eval on Batch {i} [{curr_name}]: accuracy = {acc:.4f}")

    return eval_results


def save_cross_eval_results(results: list, filename: str = "cross_eval_accuracy.json"):
    """Save cross-dataset evaluation results."""
    os.makedirs(REPORTS_PATH, exist_ok=True)
    path = os.path.join(REPORTS_PATH, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Cross-eval results saved -> {path}")