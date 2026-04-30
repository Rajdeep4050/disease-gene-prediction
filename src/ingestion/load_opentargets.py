"""
Load Open Targets gene-disease association data from parquet files (sampled version).
"""

import pandas as pd
import os
from src.config.constants import OPENTARGETS_SCORE_THRESHOLD


def load_opentargets_data(path: str, sample_frac: float = 0.02, max_files: int = 5) -> pd.DataFrame:
    """
    Load and sample Open Targets parquet dataset.

    Args:
        path (str): Folder containing parquet files
        sample_frac (float): Fraction of rows to sample from each file
        max_files (int): Number of parquet files to read (limits memory usage)

    Returns:
        pd.DataFrame: Filtered gene-disease associations
    """

    # Get parquet files
    files = sorted([f for f in os.listdir(path) if f.endswith(".parquet")])

    if not files:
        raise ValueError("No parquet files found in the specified directory")

    dfs = []

    print(f"Reading {min(len(files), max_files)} parquet files (sampled)...")

    for file in files[:max_files]:
        file_path = os.path.join(path, file)

        df = pd.read_parquet(file_path)

        # ✅ Validate schema (based on your actual output)
        required_cols = {"targetId", "diseaseId", "associationScore"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Missing expected columns in {file}: {df.columns}")

        # Sample to reduce memory usage
        df = df.sample(frac=sample_frac, random_state=42)

        dfs.append(df)

    # Combine all sampled data
    df = pd.concat(dfs, ignore_index=True)

    # Select only required columns
    df = df[["targetId", "diseaseId", "associationScore"]]

    # Rename for consistency across pipeline
    df = df.rename(columns={"associationScore": "score"})

    # Apply threshold filtering
    df = df[df["score"] > OPENTARGETS_SCORE_THRESHOLD]

    df = df.reset_index(drop=True)

    return df