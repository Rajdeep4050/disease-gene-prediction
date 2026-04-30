"""
Load and process STRING protein interaction data.
"""

import pandas as pd
from src.config.constants import STRING_SCORE_THRESHOLD


def load_string_data(path: str) -> pd.DataFrame:
    """
    Load STRING interaction data.

    Args:
        path: Path to STRING dataset

    Returns:
        Filtered interaction DataFrame
    """

    df = pd.read_csv(path, sep=" ")

    expected_cols = {"protein1", "protein2", "combined_score"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"Missing columns in STRING data: {df.columns}")

    df = df[["protein1", "protein2", "combined_score"]]

    # Filter high-confidence edges
    df = df[df["combined_score"] > STRING_SCORE_THRESHOLD]

    df = df.reset_index(drop=True)

    return df