"""
shap_analysis.py
----------------
SHAP feature importance analysis under data drift.

Answers: when a model is tested on a drifted batch, which features
are driving its predictions, and how do they differ from the reference?

Explainer selection:
  random_forest, xgboost, decision_tree -> TreeExplainer  (fast, exact)
  logistic, svm                         -> LinearExplainer (fast, exact)

Run from src/:
  pip install shap matplotlib
  python shap_analysis.py

Output (in artifacts/reports/):
  shap_comparison_{model}.png     side-by-side bar chart
  shap_values_{model}.json        mean |SHAP| per feature
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    PROCESSED_DATA_PATH, PROCESSED_FILENAME,
    TARGET_COLUMN, MODELS_PATH, REPORTS_PATH,
    RANDOM_STATE, SHAP_SAMPLE_SIZE, SHAP_TOP_N_FEATURES, DEBUG,
)
from models import get_shap_explainer_type
from utils import get_logger

logger = get_logger("shap_analysis")

SHAP_MODELS         = ["random_forest", "xgboost", "decision_tree", "logistic", "svm"]
REFERENCE_BATCH_IDX = 0   # Monday
DRIFTED_BATCH_IDX   = 7   # Friday DDoS — highest drift


def load_processed_data() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DATA_PATH, PROCESSED_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data not found: {path}")
    logger.info(f"Loading: {path}")
    
    if DEBUG:
        logger.info("DEBUG mode: Chunk-loading to prevent memory crash...")
        chunk_list = []
        for chunk in pd.read_csv(path, chunksize=50_000, low_memory=False):
            # Keep 10% of each chunk to maintain distribution without crashing RAM
            chunk_list.append(chunk.sample(frac=0.10, random_state=RANDOM_STATE))
        return pd.concat(chunk_list, ignore_index=True)
    else:
        return pd.read_csv(path, low_memory=False)


def get_batches(df: pd.DataFrame):
    from config import RAW_FILES
    ref_file     = RAW_FILES[REFERENCE_BATCH_IDX]
    drifted_file = RAW_FILES[DRIFTED_BATCH_IDX]
    ref     = df[df["source_file"] == ref_file]
    drifted = df[df["source_file"] == drifted_file]
    logger.info(f"Reference : {ref_file} ({len(ref)} rows)")
    logger.info(f"Drifted   : {drifted_file} ({len(drifted)} rows)")
    return ref, drifted, ref_file, drifted_file


def _prep(batch, feature_cols, needs_scaling, scaler):
    """Sample and optionally scale a batch for SHAP computation."""
    s = batch.sample(min(SHAP_SAMPLE_SIZE, len(batch)), random_state=RANDOM_STATE)
    X = s[feature_cols].values
    if needs_scaling and scaler is not None:
        X = scaler.transform(s[feature_cols])
    return X


def run_shap_for_model(model_type, ref_batch, drifted_batch, ref_name, drifted_name):
    try:
        import shap
    except ImportError:
        logger.error("SHAP not installed. Run: pip install shap")
        return

    model_path = os.path.join(MODELS_PATH, f"{model_type}_v1.pkl")
    if not os.path.exists(model_path):
        logger.warning(f"Model not found: {model_path} — train it first with main.py")
        return

    logger.info(f"\n--- SHAP: {model_type} ---")
    model  = joblib.load(model_path)

    scaler_path   = os.path.join(os.path.dirname(MODELS_PATH), "scalers", "scaler_v1.pkl")
    scaler        = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    needs_scaling = model_type in {"svm", "logistic"}
    exp_type      = get_shap_explainer_type(model_type)

    feature_cols = [
        c for c in ref_batch.columns
        if c not in [TARGET_COLUMN, "source_file"]
        and pd.api.types.is_numeric_dtype(ref_batch[c])
    ]

    X_ref     = _prep(ref_batch,     feature_cols, needs_scaling, scaler)
    X_drifted = _prep(drifted_batch, feature_cols, needs_scaling, scaler)

    try:
        if exp_type == "tree":
            explainer    = shap.TreeExplainer(model)
            sv_ref       = explainer.shap_values(X_ref)
            sv_drifted   = explainer.shap_values(X_drifted)
            
        elif exp_type == "linear":
            explainer    = shap.LinearExplainer(model, X_ref[:100])
            sv_ref       = explainer.shap_values(X_ref)
            sv_drifted   = explainer.shap_values(X_drifted)
            
        else:
            logger.warning(f"No SHAP explainer for {model_type}")
            return

        # ── FIX: Safely extract Class 1 SHAP values for any SHAP version ──
        if isinstance(sv_ref, list):
            # Older SHAP versions return a list
            sv_ref     = sv_ref[1]
            sv_drifted = sv_drifted[1]
        elif len(sv_ref.shape) == 3:
            # Newer SHAP versions return a 3D array (samples, features, classes)
            sv_ref     = sv_ref[:, :, 1]
            sv_drifted = sv_drifted[:, :, 1]

    except Exception as e:
        logger.error(f"SHAP failed for {model_type}: {e}")
        return

    mean_ref     = np.abs(sv_ref).mean(axis=0)
    mean_drifted = np.abs(sv_drifted).mean(axis=0)

    top_idx      = np.argsort(mean_ref)[-SHAP_TOP_N_FEATURES:][::-1]
    top_features = [feature_cols[i] for i in top_idx]
    ref_vals     = [round(float(mean_ref[i]),     6) for i in top_idx]
    drift_vals   = [round(float(mean_drifted[i]), 6) for i in top_idx]

    # ── Side-by-side plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f"SHAP Feature Importance Under Drift  |  Model: {model_type}\n"
        f"Left: Reference (Monday)   Right: Drifted (Friday DDoS)",
        fontsize=12
    )
    for ax, vals, title, colour in [
        (axes[0], ref_vals,   f"Reference\n{ref_name[:45]}",     "#2E86AB"),
        (axes[1], drift_vals, f"Drifted\n{drifted_name[:45]}", "#E84855"),
    ]:
        y = range(len(top_features))
        ax.barh(list(y), vals, color=colour, alpha=0.85, edgecolor="none")
        ax.set_yticks(list(y))
        ax.set_yticklabels(top_features, fontsize=8)
        ax.set_xlabel("Mean |SHAP value|", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    os.makedirs(REPORTS_PATH, exist_ok=True)
    plot_path = os.path.join(REPORTS_PATH, f"shap_comparison_{model_type}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved -> {plot_path}")

    # ── JSON summary ───────────────────────────────────────────────────────
    summary = {
        "model"          : model_type,
        "reference_batch": ref_name,
        "drifted_batch"  : drifted_name,
        "top_features"   : top_features,
        "reference_mean_abs_shap": dict(zip(top_features, ref_vals)),
        "drifted_mean_abs_shap"  : dict(zip(top_features, drift_vals)),
    }
    json_path = os.path.join(REPORTS_PATH, f"shap_values_{model_type}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"JSON saved -> {json_path}")

    # ── Log top 5 most shifted features ───────────────────────────────────
    logger.info("  Top 5 features — reference vs drifted importance:")
    for feat, rv, dv in zip(top_features[:5], ref_vals[:5], drift_vals[:5]):
        change = dv - rv
        logger.info(
            f"    {feat:<38} ref={rv:.4f}  drift={dv:.4f}  "
            f"delta={change:+.4f}"
        )


def run_shap_analysis():
    logger.info("=== SHAP Analysis Started ===")
    df = load_processed_data()
    ref_batch, drifted_batch, ref_name, drifted_name = get_batches(df)

    for model_type in SHAP_MODELS:
        run_shap_for_model(model_type, ref_batch, drifted_batch,
                           ref_name, drifted_name)

    logger.info("\n=== SHAP Analysis Complete ===")
    logger.info(f"All outputs in: {REPORTS_PATH}")


if __name__ == "__main__":
    run_shap_analysis()
