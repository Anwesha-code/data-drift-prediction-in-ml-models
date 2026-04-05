"""
preprocessing.py
----------------
Data loading and cleaning pipeline for CICIDS 2017.

Memory strategy — definitive, architecture-level fix
=====================================================
The fundamental constraint is that this machine cannot hold the full
2.8M-row combined dataset (roughly 1.8 GB as float64, ~1.1 GB as float32)
in memory in one contiguous allocation. Every approach that eventually calls
pd.concat() or pd.read_csv() on the full combined data will fail on this
hardware, regardless of how clean the chunks are.

The correct solution is to NEVER combine all 8 files into a single
in-memory DataFrame at any point during preprocessing:

  Stage 1 — Per-file streaming to disk
    Each raw CSV is read in 20,000-row chunks. Every chunk is cleaned
    (inf replacement via NumPy, NaN drop, float32 downcast) and written
    to an individual interim CSV at data/interim/<filename>.csv.
    Peak memory: one 20,000-row chunk (~6 MB).

  Stage 2 — Feature selector fitted on Monday only
    The Monday interim file (~530k rows, ~200 MB) is loaded in isolation.
    VarianceThreshold and correlation filtering are fitted on this file.
    The resulting list of columns to keep is recorded.
    Monday is used as the reference because feature selection should be
    based on the training/reference distribution only — using all 8 days
    would allow test-day information to influence feature selection, which
    is a form of data leakage.
    Peak memory: Monday interim file (~200 MB).

  Stage 3 — Per-file transform and append to processed CSV
    Each interim file is loaded one at a time, the pre-computed column
    list is applied, unnecessary columns are dropped, labels are encoded,
    and the result is appended to data/processed/processed_data_v1.csv.
    Peak memory: one interim file (~100–470 MB).

The final processed CSV is then read back one more time during training
(unavoidable — sklearn needs the full matrix), but that read is a single
fresh pd.read_csv() which Pandas handles without triggering block
consolidation because it builds column blocks from scratch.

If that final read still fails due to hardware limitations, set
DEBUG = True in config.py to work with a 100,000-row sample, or increase
your system's virtual memory / page file on Windows.
"""

import gc
import os

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from config import *
from utils import get_logger
from feature_engineering import fit_feature_selector, apply_feature_selector

logger = get_logger("preprocessing")


# ── Interim file path helper ──────────────────────────────────────────────────

def _interim_path(filename: str) -> str:
    """Return the interim CSV path for a given raw source filename."""
    return os.path.join(INTERIM_DATA_PATH, filename + ".interim.csv")


# ── Dtype optimisation ────────────────────────────────────────────────────────

def _downcast_floats(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Convert float64 columns to float32, one column at a time.
    Per-column assignment avoids triggering Pandas block consolidation,
    which requires a large contiguous allocation across all float columns.
    """
    for col in chunk.select_dtypes(include=["float64"]).columns:
        chunk[col] = chunk[col].astype(np.float32)
    return chunk


# ── Per-chunk cleaning ────────────────────────────────────────────────────────

def _clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a single 20,000-row chunk:
      1. Strip leading/trailing whitespace from column names.
      2. Replace +/-inf with NaN using NumPy (avoids Pandas replace() path).
      3. Drop NaN rows.
      4. Downcast float64 to float32.
    """
    chunk.columns = chunk.columns.str.strip()

    float_cols = chunk.select_dtypes(include=["float32", "float64"]).columns
    if len(float_cols):
        arr = chunk[float_cols].to_numpy(dtype=np.float64, copy=True)
        arr[~np.isfinite(arr)] = np.nan
        chunk[float_cols] = arr

    chunk.dropna(inplace=True)
    chunk = _downcast_floats(chunk)
    return chunk


# ── Stage 1: stream each file to its own interim CSV ─────────────────────────

def _stream_file_to_interim(filename: str) -> int:
    """
    Read one raw CICIDS CSV in 20,000-row chunks, clean each chunk,
    and write it to data/interim/<filename>.interim.csv.

    Returns the number of rows successfully written.
    """
    filepath = os.path.join(RAW_DATA_PATH, filename)
    out_path = _interim_path(filename)

    if not os.path.exists(filepath):
        logger.warning(f"File not found, skipping: {filename}")
        return 0

    # Remove stale interim from a previous failed run.
    if os.path.exists(out_path):
        os.remove(out_path)

    rows_written = 0
    rows_removed = 0
    header_done  = False

    for chunk in pd.read_csv(filepath, chunksize=20_000, low_memory=False):
        chunk["source_file"] = filename
        before       = len(chunk)
        chunk        = _clean_chunk(chunk)
        rows_removed += before - len(chunk)

        if len(chunk) == 0:
            continue

        chunk.to_csv(out_path, mode="a", header=not header_done, index=False)
        header_done   = True
        rows_written += len(chunk)
        del chunk
        gc.collect()

    if rows_removed:
        logger.info(
            f"  [{filename[:30]}] "
            f"kept {rows_written:,} rows, removed {rows_removed:,} NaN/Inf rows"
        )
    else:
        logger.info(f"  [{filename[:30]}] kept {rows_written:,} rows")

    return rows_written


def stream_all_files_to_interim():
    """Run Stage 1 for every file in RAW_FILES."""
    os.makedirs(INTERIM_DATA_PATH, exist_ok=True)
    total = 0
    for filename in RAW_FILES:
        logger.info(f"Streaming: {filename}")
        total += _stream_file_to_interim(filename)
        gc.collect()
    logger.info(f"Stage 1 complete: {total:,} rows written to interim/")


# ── Stage 2: fit feature selector on Monday only ──────────────────────────────

def fit_selector_on_reference():
    reference_file = RAW_FILES[0]
    ref_path = _interim_path(reference_file)

    logger.info("Sampling reference data for feature selection")

    ref_df = None

    for chunk in pd.read_csv(ref_path, chunksize=20000, low_memory=False):
        chunk = chunk.sample(frac=0.2, random_state=42)

        cols_to_drop = [c for c in COLUMNS_TO_DROP if c in chunk.columns]
        chunk.drop(columns=cols_to_drop, inplace=True)

        if "source_file" in chunk.columns:
            chunk.drop(columns=["source_file"], inplace=True)

        chunk = _encode_labels(chunk)

        if ref_df is None:
            ref_df = chunk
        else:
            ref_df = pd.concat([ref_df, chunk], ignore_index=True)

        del chunk
        gc.collect()

    cols_to_keep = fit_feature_selector(ref_df)

    del ref_df
    gc.collect()

    return cols_to_keep

# ── Label encoding helper ─────────────────────────────────────────────────────

def _encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """BENIGN -> 0, all attack labels -> 1 (int8)."""
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].str.strip()
        df[TARGET_COLUMN] = (df[TARGET_COLUMN] != "BENIGN").astype(np.int8)
    return df


# ── Stage 3: transform each interim file and append to processed CSV ──────────

def build_processed_csv(cols_to_keep: list):
    processed_path = os.path.join(PROCESSED_DATA_PATH, PROCESSED_FILENAME)
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    if os.path.exists(processed_path):
        os.remove(processed_path)

    header_done   = False
    total_written = 0

    for filename in RAW_FILES:
        interim_path = _interim_path(filename)

        if not os.path.exists(interim_path):
            logger.warning(f"Interim file missing, skipping: {filename}")
            continue

        logger.info(f"Processing: {filename}")

        for chunk in pd.read_csv(interim_path, chunksize=50000, low_memory=False):

            cols_drop = [c for c in COLUMNS_TO_DROP if c in chunk.columns]
            chunk.drop(columns=cols_drop, inplace=True)

            source_col = None
            if "source_file" in chunk.columns:
                source_col = chunk["source_file"].copy()
                chunk.drop(columns=["source_file"], inplace=True)

            chunk = _encode_labels(chunk)
            chunk = apply_feature_selector(chunk, cols_to_keep)

            if source_col is not None:
                chunk["source_file"] = source_col.values

            chunk.to_csv(
                processed_path,
                mode="a",
                header=not header_done,
                index=False
            )

            header_done = True
            total_written += len(chunk)

            logger.info(
                f"  Written {len(chunk):,} rows "
                f"(total so far: {total_written:,})"
            )

            del chunk, source_col
            gc.collect()

    logger.info(f"Stage 3 complete: {total_written:,} rows in processed CSV")


# ── Constant column removal (across combined dataset) ─────────────────────────

# ── Constant column removal (corrected logic) ─────────────────────────

def _find_constant_columns() -> list:
    processed_path = os.path.join(PROCESSED_DATA_PATH, PROCESSED_FILENAME)

    unique_vals = {}

    for chunk in pd.read_csv(processed_path, chunksize=20000, low_memory=False):
        for col in chunk.columns:
            if col in ["source_file", TARGET_COLUMN]:
                continue

            vals = set(chunk[col].dropna().unique())

            if col not in unique_vals:
                unique_vals[col] = vals
            else:
                # Only track until >1 unique (optimization)
                if len(unique_vals[col]) <= 1:
                    unique_vals[col].update(vals)

        del chunk
        gc.collect()

    constant = [col for col, vals in unique_vals.items() if len(vals) <= 1]
    return constant


def _drop_constant_columns(constant_cols: list):
    """
    Remove constant columns from the processed CSV by rewriting it in
    chunks.  Overwrites the file in-place.
    """
    if not constant_cols:
        logger.info("No constant columns found across combined dataset.")
        return

    logger.info(f"Removing {len(constant_cols)} constant columns from processed CSV.")
    processed_path = os.path.join(PROCESSED_DATA_PATH, PROCESSED_FILENAME)
    tmp_path       = processed_path + ".tmp"

    header_done = False
    for chunk in pd.read_csv(processed_path, chunksize=20_000, low_memory=False):
        chunk.drop(columns=constant_cols, errors="ignore", inplace=True)
        chunk.to_csv(tmp_path, mode="a", header=not header_done, index=False)
        header_done = True
        del chunk
        gc.collect()

    os.replace(tmp_path, processed_path)
    logger.info("Constant column removal complete.")


# ── Scaling ───────────────────────────────────────────────────────────────────

def scale_features(X_train, X_test):
    """
    Fit StandardScaler on the training split only, then transform both.
    Fitted on training data exclusively to prevent test-set leakage.
    """
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    if SAVE_SCALER:
        os.makedirs(SCALERS_PATH, exist_ok=True)
        path = os.path.join(SCALERS_PATH, "scaler_v1.pkl")
        joblib.dump(scaler, path)
        logger.info(f"Scaler saved -> {path}")

    return X_train_scaled, X_test_scaled, scaler


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_preprocessing() -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    Stage 1 — Streaming write
      Each raw file is cleaned chunk-by-chunk and written to its own
      interim CSV. No combined DataFrame is ever built.

    Stage 2 — Feature selection fitted on Monday
      VarianceThreshold and correlation filter are fitted on the Monday
      interim file (~530k rows). The selected column list is saved.

    Stage 3 — Per-file transform and final CSV construction
      Each interim file is loaded, the column selection is applied, and
      the result is appended to the processed CSV.

    Stage 4 — Constant column scan (chunked)
      The processed CSV is scanned in 50,000-row chunks to find any
      globally constant columns, which are then removed.

    Returns
    -------
    pd.DataFrame
        The fully processed dataset, loaded from the final CSV.
        This is the one unavoidable full-dataset load (needed for
        train_test_split and model training in train.py).
    """
    # Stage 1
    logger.info("=== Stage 1: Streaming raw files to interim/ ===")
    stream_all_files_to_interim()

    # Stage 2
    logger.info("=== Stage 2: Fitting feature selector on Monday ===")
    cols_to_keep = fit_selector_on_reference()

    # Stage 3
    logger.info("=== Stage 3: Building processed CSV ===")
    build_processed_csv(cols_to_keep)

    # Stage 4
    logger.info("=== Stage 4: Scanning for globally constant columns ===")
    constant_cols = _find_constant_columns()
    _drop_constant_columns(constant_cols)

   # ── Final load (OOM-Safe) ─────────────────────────────────────────────────
    processed_path = os.path.join(PROCESSED_DATA_PATH, PROCESSED_FILENAME)
    logger.info(f"Loading processed dataset: {processed_path}")

    if DEBUG:
        logger.info("DEBUG mode: Chunk-loading a sample to prevent memory crash...")
        chunk_list = []
        # Read the file in small chunks so it never overwhelms your RAM
        for chunk in pd.read_csv(processed_path, chunksize=50_000, low_memory=False):
            # Keep only 5% of each chunk
            chunk_list.append(chunk.sample(frac=0.05, random_state=RANDOM_STATE))
        
        # Combine the tiny chunks into one safe DataFrame
        df = pd.concat(chunk_list, ignore_index=True)
        
        # Cap it exactly at 100,000 rows if it exceeded it
        if len(df) > 100_000:
            df = df.sample(100_000, random_state=RANDOM_STATE)
    else:
        # Warning: This requires ~2GB+ of free RAM
        df = pd.read_csv(processed_path, low_memory=False)

    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    logger.info(f"Loaded {len(df):,} rows | Memory: {mem_mb:.1f} MB")

    if "source_file" not in df.columns:
        raise ValueError(
            "source_file column is missing from the processed dataset. "
            "Check that RAW_FILES in config.py matches actual filenames."
        )

    return df
