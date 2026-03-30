# KL divergence

# JS divergence

# drift detection logic
import pandas as pd
from utils import get_logger

logger = get_logger("drift")


def split_by_source(df: pd.DataFrame):
    batches = []

    for name, group in df.groupby("source_file"):
        logger.info(f"{name}: {len(group)} rows")
        batches.append(group)

    return batches


def check_distribution(batches):
    for i, batch in enumerate(batches):
        dist = batch["Label"].value_counts(normalize=True)
        logger.info(f"Batch {i} distribution:\n{dist}")