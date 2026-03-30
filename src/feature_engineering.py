import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from config import LOW_VARIANCE_THRESHOLD, CORRELATION_THRESHOLD
from utils import get_logger

logger = get_logger("feature_engineering")


def remove_low_variance(df: pd.DataFrame) -> pd.DataFrame:
    selector = VarianceThreshold(threshold=LOW_VARIANCE_THRESHOLD)
    numeric_df = df.select_dtypes(include=[np.number])

    selector.fit(numeric_df)
    cols = numeric_df.columns[selector.get_support()]

    logger.info(f"Low variance removed: {len(numeric_df.columns) - len(cols)} features")

    return df[cols.tolist() + [c for c in df.columns if c not in numeric_df.columns]]


def remove_high_correlation(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])

    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > CORRELATION_THRESHOLD)]

    logger.info(f"Highly correlated dropped: {len(to_drop)} features")

    return df.drop(columns=to_drop, errors="ignore")


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_low_variance(df)
    df = remove_high_correlation(df)
    return df