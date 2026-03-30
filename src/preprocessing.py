import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import joblib

from config import *
from utils import get_logger
from feature_engineering import apply_feature_engineering  # NEW

logger = get_logger("preprocessing")


def load_all_raw_files() -> pd.DataFrame:
    """Load and concatenate all CSV files using chunking (memory-safe)."""
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

        df_file = pd.concat(chunk_list, ignore_index=True)
        dfs.append(df_file)

    combined = pd.concat(dfs, ignore_index=True)

    logger.info(f"Total records loaded: {len(combined)}")

    return combined

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle infinities, missing values, and column issues."""
    initial = len(df)

    # Normalize column names
    df.columns = df.columns.str.strip()

    # Replace inf and drop NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Remove constant columns EXCEPT source_file
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1 and col != "source_file"]
    df.drop(columns=constant_cols, inplace=True)

    logger.info(f"Removed {initial - len(df)} rows with NaN/Inf values.")
    logger.info(f"Constant columns removed: {len(constant_cols)}")

    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are not useful for modelling."""
    cols_present = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df.drop(columns=cols_present, inplace=True)
    logger.info(f"Dropped columns: {cols_present}")
    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Encode target: BENIGN → 0, everything else → 1."""
    df[TARGET_COLUMN] = df[TARGET_COLUMN].str.strip()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(lambda x: 0 if x == "BENIGN" else 1)

    # Normalized distribution (important for drift)
    logger.info(f"Label distribution:\n{df[TARGET_COLUMN].value_counts(normalize=True)}")
    return df


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if SAVE_SCALER:
        os.makedirs(SCALERS_PATH, exist_ok=True)
        scaler_path = os.path.join(SCALERS_PATH, "scaler_v1.pkl")
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved -> {scaler_path}")

    return X_train_scaled, X_test_scaled, scaler


def save_processed(df: pd.DataFrame):
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    path = os.path.join(PROCESSED_DATA_PATH, PROCESSED_FILENAME)
    df.to_csv(path, index=False)
    logger.info(f"Processed dataset saved -> {path}")


def run_preprocessing() -> pd.DataFrame:
    """Full preprocessing pipeline."""
    df = load_all_raw_files()

    # DEBUG sampling
    if DEBUG:
        df = df.sample(100000, random_state=RANDOM_STATE)

    df = clean_data(df)
    df = drop_unnecessary_columns(df)
    df = encode_labels(df)

    # Ensure source_file exists
    if "source_file" not in df.columns:
        raise ValueError("source_file column missing after loading")

    # Save before feature engineering
    source_col = df["source_file"].copy()

    # Apply feature engineering ONCE
    df = apply_feature_engineering(df)

    # Restore source_file
    df["source_file"] = source_col

    # Save processed data
    if SAVE_PROCESSED_DATA:
        save_processed(df)

    return df