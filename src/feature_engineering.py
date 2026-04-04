"""
feature_engineering.py
-----------------------
Feature selection for the CICIDS 2017 pipeline.

Two-phase API (fit then transform)
====================================
The original single-pass apply_feature_engineering(df) required the full
2.8M-row combined dataset in memory so that variance and correlation could
be computed globally. On machines with limited RAM this caused an OOM crash
during the combined CSV read-back.

The new API separates the concerns:

  fit_feature_selector(df)
      Computes which columns to drop based on low variance and high
      correlation. Called ONCE on the Monday reference batch (~530k rows),
      which comfortably fits in memory. Returns a plain dict of column
      names to keep.

  apply_feature_selector(df, cols_to_keep)
      Drops every column not in cols_to_keep. Called on each day's
      interim file individually so only one file is in memory at a time.

  apply_feature_engineering(df)
      Original single-pass function retained for backward compatibility
      with any code that calls it directly (e.g. notebooks). It internally
      calls fit_feature_selector + apply_feature_selector, so it still
      works correctly when the full dataset is available.

Using Monday as the reference for feature selection is scientifically
sound: feature selection should be based on the training/reference
distribution only, not on test-day data, to avoid any form of leakage.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from config import LOW_VARIANCE_THRESHOLD, CORRELATION_THRESHOLD, TARGET_COLUMN
from utils import get_logger

logger = get_logger("feature_engineering")


# ── Fit phase ─────────────────────────────────────────────────────────────────

def fit_feature_selector(df: pd.DataFrame) -> list:
    """
    Compute the list of numeric columns to retain after variance and
    correlation filtering.

    This function is intended to be called on the Monday reference batch
    only. The returned column list is then passed to apply_feature_selector()
    for every subsequent day's data, ensuring no test-day information
    influences which features are selected.

    Parameters
    ----------
    df : DataFrame
        Reference batch (Monday), excluding source_file. The Label /
        TARGET_COLUMN may be present; it is excluded from selection.

    Returns
    -------
    list of str
        Column names to keep (numeric features that passed both filters,
        plus any non-numeric columns that were present).
    """
    # Work only on numeric columns for the filters.
    # Non-numeric columns (Label, source_file if present) are preserved as-is.
    numeric_df = df.select_dtypes(include=[np.number])

    # ── Step 1: variance filter ────────────────────────────────────────────
    selector = VarianceThreshold(threshold=LOW_VARIANCE_THRESHOLD)
    selector.fit(numeric_df)
    after_variance = numeric_df.columns[selector.get_support()].tolist()
    n_removed_var  = len(numeric_df.columns) - len(after_variance)
    logger.info(f"Low variance removed: {n_removed_var} features")

    # ── Step 2: correlation filter ─────────────────────────────────────────
    corr_df     = numeric_df[after_variance]
    corr_matrix = corr_df.corr().abs()
    upper       = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if any(upper[col] > CORRELATION_THRESHOLD)]
    after_corr = [c for c in after_variance if c not in to_drop]
    logger.info(f"Highly correlated dropped: {len(to_drop)} features")
    logger.info(f"Features retained after selection: {len(after_corr)}")

    # Include non-numeric columns (e.g. Label) in the keep list so they
    # are not accidentally dropped when apply_feature_selector() runs.
    non_numeric = [c for c in df.columns if c not in numeric_df.columns]
    cols_to_keep = after_corr + non_numeric

    return cols_to_keep


# ── Transform phase ────────────────────────────────────────────────────────────

def apply_feature_selector(df: pd.DataFrame, cols_to_keep: list) -> pd.DataFrame:
    """
    Drop every column not present in cols_to_keep.

    Handles the case where a day's file may be missing some columns or
    have extra columns relative to the reference batch by intersecting
    rather than requiring an exact match.

    Parameters
    ----------
    df         : DataFrame for one day.
    cols_to_keep : Column list returned by fit_feature_selector().

    Returns
    -------
    Filtered DataFrame with only the selected columns (in original order).
    """
    present = [c for c in cols_to_keep if c in df.columns]
    dropped = len(df.columns) - len(present)
    if dropped:
        logger.info(f"  apply_feature_selector: dropped {dropped} columns")
    return df[present]


# ── Combined single-pass (backward-compatible) ────────────────────────────────

def fit_feature_selector(df: pd.DataFrame) -> list:
    """
    Compute the list of numeric columns to retain after variance and
    correlation filtering.
    """
    # Work only on numeric columns for the filters.
    numeric_df = df.select_dtypes(include=[np.number])

    # ── FIX: Protect the Target Column from the Variance Filter ──
    if TARGET_COLUMN in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[TARGET_COLUMN])

    # ── Step 1: variance filter ────────────────────────────────────────────
    selector = VarianceThreshold(threshold=LOW_VARIANCE_THRESHOLD)
    selector.fit(numeric_df)
    after_variance = numeric_df.columns[selector.get_support()].tolist()
    n_removed_var  = len(numeric_df.columns) - len(after_variance)
    logger.info(f"Low variance removed: {n_removed_var} features")

    # ── Step 2: correlation filter ─────────────────────────────────────────
    corr_df     = numeric_df[after_variance]
    corr_matrix = corr_df.corr().abs()
    upper       = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if any(upper[col] > CORRELATION_THRESHOLD)]
    after_corr = [c for c in after_variance if c not in to_drop]
    logger.info(f"Highly correlated dropped: {len(to_drop)} features")
    logger.info(f"Features retained after selection: {len(after_corr)}")

    # Include non-numeric columns (e.g. Label) in the keep list so they
    # are not accidentally dropped when apply_feature_selector() runs.
    non_numeric = [c for c in df.columns if c not in numeric_df.columns]
    cols_to_keep = after_corr + non_numeric

    return cols_to_keep
