"""
preprocessing.py
----------------
Data loading and cleaning pipeline for CICIDS 2017.

Memory strategy (fixes ArrayMemoryError):
  - Inf/NaN replacement and NaN dropping happen PER FILE inside the
    loading loop, before concatenation. This avoids calling df.replace()
    on the 2.8M-row combined DataFrame, which triggers an expensive
    pandas block-consolidation step that requires 292+ MB contiguous.
  - Only float64 columns are downcast to float32 (saving ~50% on floats).
    Integer columns are left as-is — downcasting int64->int32 created the
    consolidation problem by producing many same-dtype blocks.
  - gc.collect() is called after each file to release chunk memory early.
"""

import gc
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import joblib

from config import *
from utils import get_logger
from feature_engineering import apply_feature_engineering

logger = get_logger("preprocessing")


# ── Dtype optimisation ────────────────────────────────────────────────────────

def _downcast_floats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast float64 columns to float32 only.
    Saves ~50% memory on float columns without touching integers.

    We deliberately avoid downcasting integers because it causes pandas to
    create many same-dtype blocks that later trigger expensive consolidation
    during df.replace() — which was the original ArrayMemoryError.
    """
    float_cols = df.select_dtypes(include=["float64"]).columns
    for col in float_cols:
        df[col] = df[col].astype(np.float32)
    return df


# ── Per-file cleaning (run BEFORE concat) ─────────────────────────────────────

def _clean_file(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Clean a single file's DataFrame before it is added to the combined list.

    Steps:
      1. Strip column name whitespace (CICIDS has leading spaces)
      2. Replace inf/-inf with NaN
      3. Drop NaN rows
      4. Downcast float64 -> float32

    Doing this per file means the expensive replace() and dropna() operations
    run on ~200-700k rows at a time rather than on 2.8M rows all at once.
    """
    initial = len(df)

    # Normalise column names
    df.columns = df.columns.str.strip()

    # Replace inf values — only in float columns (where inf can occur)
    float_cols = df.select_dtypes(include=["float32", "float64"]).columns
    df[float_cols] = df[float_cols].replace([np.inf, -np.inf], np.nan)

    # Drop any NaN rows
    df.dropna(inplace=True)

    removed = initial - len(df)
    if removed > 0:
        logger.info(f"  [{filename[:30]}] Removed {removed:,} NaN/Inf rows")

    # Downcast floats only
    df = _downcast_floats(df)

    return df


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_all_raw_files() -> pd.DataFrame:
    """
    Load all CICIDS CSV files, clean each one individually, then concatenate.

    Per-file cleaning avoids:
      - Large replace() calls on the full 2.8M-row DataFrame
      - Pandas block consolidation that triggers ArrayMemoryError

    Memory log is printed after each file so you can track progress.
    """
    dfs = []

    for filename in RAW_FILES:
        filepath = os.path.join(RAW_DATA_PATH, filename)

        if not os.path.exists(filepath):
            logger.warning(f"File not found, skipping: {filename}")
            continue

        logger.info(f"Loading (chunked): {filename}")
        chunk_list = []

        for chunk in pd.read_csv(filepath, chunksize=20000, low_memory=False):
            chunk["source_file"] = filename
            chunk_list.append(chunk)

        # Concatenate this file's chunks
        df_file = pd.concat(chunk_list, ignore_index=True)
        del chunk_list
        gc.collect()

        # Clean this file before adding to the list
        df_file = _clean_file(df_file, filename)

        mem_mb = df_file.memory_usage(deep=True).sum() / 1e6
        logger.info(f"  Loaded {len(df_file):,} rows | Memory: {mem_mb:.1f} MB")

        dfs.append(df_file)

    logger.info(f"Concatenating {len(dfs)} files...")
    combined = pd.concat(dfs, ignore_index=True)

    del dfs
    gc.collect()

    total_mb = combined.memory_usage(deep=True).sum() / 1e6
    logger.info(f"Total records: {len(combined):,} | Total memory: {total_mb:.1f} MB")
    return combined


# ── Post-concat Cleaning ──────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-concatenation cleaning.
    Inf/NaN removal was already done per file — this step handles only
    the two remaining tasks: constant column removal and logging.
    """
    # Remove constant columns (zero variance across the full dataset)
    # These cannot be detected reliably per-file (a column may be constant
    # in one file but not another).
    constant_cols = [
        col for col in df.columns
        if col != "source_file" and df[col].nunique() <= 1
    ]
    if constant_cols:
        df.drop(columns=constant_cols, inplace=True)
        logger.info(f"Constant columns removed: {len(constant_cols)}")
    else:
        logger.info("No constant columns found.")

    gc.collect()
    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier/metadata columns not useful for modelling."""
    cols_present = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df.drop(columns=cols_present, inplace=True)
    logger.info(f"Dropped columns: {cols_present}")
    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Encode target: BENIGN -> 0, everything else -> 1."""
    df[TARGET_COLUMN] = df[TARGET_COLUMN].str.strip()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(
        lambda x: 0 if x == "BENIGN" else 1
    ).astype(np.int8)

    logger.info(
        f"Label distribution:\n"
        f"{df[TARGET_COLUMN].value_counts(normalize=True)}"
    )
    return df


# ── Scaling ───────────────────────────────────────────────────────────────────

def scale_features(X_train, X_test):
    """Fit StandardScaler on training split, transform both splits."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    if SAVE_SCALER:
        os.makedirs(SCALERS_PATH, exist_ok=True)
        scaler_path = os.path.join(SCALERS_PATH, "scaler_v1.pkl")
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved -> {scaler_path}")

    return X_train_scaled, X_test_scaled, scaler


# ── Save ──────────────────────────────────────────────────────────────────────

def save_processed(df: pd.DataFrame):
    """Save cleaned dataset to data/processed/ for all subsequent experiments."""
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    path = os.path.join(PROCESSED_DATA_PATH, PROCESSED_FILENAME)
    df.to_csv(path, index=False)
    logger.info(f"Processed dataset saved -> {path}")


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def run_preprocessing() -> pd.DataFrame:
    """
    Full preprocessing pipeline. Returns a clean DataFrame ready for training.

    Order of operations:
      1. Load all 8 raw CSV files with chunking
         (inf/NaN cleaning + float downcasting done per file)
      2. Concatenate into one combined DataFrame
      3. Remove any remaining constant columns (requires full dataset)
      4. Drop identifier columns (Flow ID, Source IP, etc.)
      5. Binary-encode labels (BENIGN=0, attack=1)
      6. Feature engineering (variance filter + correlation filter)
      7. Save to data/processed/processed_data_v1.csv
    """
    df = load_all_raw_files()

    if DEBUG:
        df = df.sample(100000, random_state=RANDOM_STATE)
        logger.info("DEBUG mode: using 100,000 row sample")

    df = clean_data(df)              # constant column removal only
    df = drop_unnecessary_columns(df)
    df = encode_labels(df)

    if "source_file" not in df.columns:
        raise ValueError("source_file column missing after loading")

    # Preserve source_file across feature engineering
    source_col = df["source_file"].copy()
    df = apply_feature_engineering(df)
    gc.collect()
    df["source_file"] = source_col

    if SAVE_PROCESSED_DATA:
        save_processed(df)

    return df
