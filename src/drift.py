# KL divergence

# JS divergence

# drift detection logic
import pandas as pd

def split_by_source(df: pd.DataFrame):
    """Each source file becomes a time batch"""
    return [group for _, group in df.groupby("source_file")]