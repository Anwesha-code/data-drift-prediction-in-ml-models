from preprocessing import run_preprocessing
from drift import split_by_source, check_distribution

df = run_preprocessing()

batches = split_by_source(df)
check_distribution(batches)