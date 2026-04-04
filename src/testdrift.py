"""
testdrift.py
------------
Main drift experiment runner.

Runs in this order:
  1. Load processed data
  2. Split into 8 temporal batches
  3. Check label distributions
  4. Fixed-reference drift  (Monday vs all)  -> drift_scores.json
  5. Rolling-window drift   (consecutive)    -> rolling_drift_scores.json
  6. Multi-model cross-dataset evaluation    -> cross_eval_accuracy.json
  7. Drift alert check                       -> drift_alerts.json (if needed)
  8. Print summary table

Run from src/:
  python testdrift.py
"""

import os
import pandas as pd
from config import (
    PROCESSED_DATA_PATH, PROCESSED_FILENAME,
    TARGET_COLUMN, MODELS_PATH, REPORTS_PATH,
    RANDOM_STATE, SHAP_SAMPLE_SIZE, SHAP_TOP_N_FEATURES,
    DEBUG  
)
from drift import (
    split_by_source,
    check_distribution,
    compute_drift,
    save_drift_results,
    compute_rolling_drift,
    save_rolling_drift_results,
    cross_dataset_evaluation,
    save_cross_eval_results,
    check_drift_alerts,
)
from utils import get_logger

logger = get_logger("testdrift")


def run_drift_experiment():
    logger.info("=== Drift Experiment Started ===")

   
   # ── 1. Load processed data ─────────────────────────────────────────────
    processed_path = os.path.join(PROCESSED_DATA_PATH, PROCESSED_FILENAME)
    if not os.path.exists(processed_path):
        logger.error(f"Processed data not found: {processed_path}")
        logger.error("Run main.py first.")
        return

    logger.info(f"Loading: {processed_path}")
    
    # NEW OOM-SAFE LOADING LOGIC
    if DEBUG:
        logger.info("DEBUG mode: Chunk-loading to prevent memory crash...")
        chunk_list = []
        for chunk in pd.read_csv(processed_path, chunksize=50_000, low_memory=False):
            # Keep 10% of each chunk to maintain temporal distribution
            chunk_list.append(chunk.sample(frac=0.10, random_state=RANDOM_STATE))
        df = pd.concat(chunk_list, ignore_index=True)
    else:
        df = pd.read_csv(processed_path, low_memory=False)

    logger.info(f"Total rows loaded: {len(df)}")

    # ── 2. Split into batches ──────────────────────────────────────────────
    batches = split_by_source(df)
    logger.info(f"Total batches found: {len(batches)}")

    if len(batches) < 2:
        logger.warning(
            "Only 1 batch found. Make sure all 8 CICIDS files are in "
            "RAW_FILES in config.py and preprocessing has been re-run."
        )
        return

    # ── 3. Distribution check ──────────────────────────────────────────────
    logger.info("\n--- Label Distribution per Batch ---")
    check_distribution(batches)

    # ── 4. Fixed-reference drift (Monday vs all) ───────────────────────────
    logger.info("\n--- Fixed Reference Drift (Monday vs all batches) ---")
    drift_results = compute_drift(batches)
    save_drift_results(drift_results)

    # ── 5. Rolling-window drift (consecutive batches) ──────────────────────
    if USE_ROLLING_REFERENCE:
        rolling_results = compute_rolling_drift(batches)
        save_rolling_drift_results(rolling_results)

    # ── 6. Multi-model cross-dataset evaluation ────────────────────────────
    logger.info(
        f"\n--- Multi-Model Cross-Dataset Evaluation ---\n"
        f"Models: {CROSS_EVAL_MODELS}"
    )
    eval_results = cross_dataset_evaluation(batches, model_types=CROSS_EVAL_MODELS)
    save_cross_eval_results(eval_results)

    # ── 7. Drift alerts ────────────────────────────────────────────────────
    logger.info("\n--- Checking Drift Alerts ---")
    alerts = check_drift_alerts(eval_results, drift_results)

    # ── 8. Summary table ───────────────────────────────────────────────────
    logger.info("\n=== CROSS-DATASET EVALUATION SUMMARY ===")

    # Collect unique batches in order
    batch_order = []
    seen = set()
    for r in eval_results:
        if r["batch"] not in seen:
            batch_order.append(r["batch"])
            seen.add(r["batch"])

    # Build per-batch per-model accuracy dict
    acc_table: dict = {b: {} for b in batch_order}
    for r in eval_results:
        acc_table[r["batch"]][r["model"]] = r["accuracy"]

    models_seen = CROSS_EVAL_MODELS
    col_w       = 12

    header = f"{'Batch':<52}" + "".join(f"{m:>{col_w}}" for m in models_seen)
    logger.info(header)
    logger.info("-" * len(header))

    for batch in batch_order:
        row = f"{batch:<52}"
        for m in models_seen:
            val = acc_table[batch].get(m, "N/A")
            row += f"{str(val):>{col_w}}"
        logger.info(row)

    logger.info(f"\nDrift scores         -> artifacts/reports/drift_scores.json")
    logger.info(f"Rolling drift        -> artifacts/reports/rolling_drift_scores.json")
    logger.info(f"Cross-eval accuracy  -> artifacts/reports/cross_eval_accuracy.json")
    if alerts:
        logger.info(f"Drift alerts         -> artifacts/reports/drift_alerts.json  ({len(alerts)} alerts)")
    logger.info("=== Drift Experiment Complete ===")


if __name__ == "__main__":
    run_drift_experiment()
