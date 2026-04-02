"""
testdrift.py
------------
Run this after your baseline models are trained.
This script is your core research experiment runner.

What it does:
  1. Loads processed data (must exist in data/processed/)
  2. Splits into time-ordered batches (one per CSV file)
  3. Shows label distribution per batch
  4. Computes KL and JS divergence (drift scores)
  5. Trains on Batch 0, evaluates on all batches (cross-dataset experiment)
  6. Saves all results to artifacts/reports/

Run from src/:
  python testdrift.py
"""

import os
import pandas as pd
from config import PROCESSED_DATA_PATH, PROCESSED_FILENAME
from drift import (
    split_by_source,
    check_distribution,
    compute_drift,
    save_drift_results,
    cross_dataset_evaluation,
    save_cross_eval_results,
)
from utils import get_logger

logger = get_logger("testdrift")


def run_drift_experiment():
    logger.info("=== Drift Experiment Started ===")

    # ── 1. Load processed data ─────────────────────────────────────────────
    processed_path = os.path.join(PROCESSED_DATA_PATH, PROCESSED_FILENAME)

    if not os.path.exists(processed_path):
        logger.error(f"Processed data not found at: {processed_path}")
        logger.error("Run main.py first to preprocess the data.")
        return

    logger.info(f"Loading: {processed_path}")
    df = pd.read_csv(processed_path)
    logger.info(f"Total rows loaded: {len(df)}")

    # ── 2. Split into batches ──────────────────────────────────────────────
    batches = split_by_source(df)
    logger.info(f"Total batches found: {len(batches)}")

    if len(batches) < 2:
        logger.warning(
            "Only 1 batch found. Drift analysis needs multiple files. "
            "Make sure all 8 CICIDS files are in RAW_FILES in config.py "
            "and preprocessing has been re-run."
        )
        return

    # ── 3. Distribution check ──────────────────────────────────────────────
    logger.info("\n--- Label Distribution per Batch ---")
    check_distribution(batches)

    # ── 4. Compute drift scores ────────────────────────────────────────────
    logger.info("\n--- Computing Drift Scores ---")
    drift_results = compute_drift(batches)
    save_drift_results(drift_results)

    # ── 5. Cross-dataset evaluation ────────────────────────────────────────
    logger.info("\n--- Cross-Dataset Evaluation (Core Research Experiment) ---")
    eval_results = cross_dataset_evaluation(batches)
    save_cross_eval_results(eval_results)

    # ── 6. Summary printout ────────────────────────────────────────────────
    logger.info("\n=== SUMMARY ===")
    logger.info(f"{'Batch':<55} {'Accuracy':>10} {'Note'}")
    logger.info("-" * 85)
    for r in eval_results:
        logger.info(f"{r['batch']:<55} {r['accuracy']:>10.4f}  {r['note']}")

    logger.info("\nDrift scores saved to artifacts/reports/drift_scores.json")
    logger.info("Accuracy results saved to artifacts/reports/cross_eval_accuracy.json")
    logger.info("=== Drift Experiment Complete ===")


if __name__ == "__main__":
    run_drift_experiment()