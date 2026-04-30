"""
Create gene-level labels from Open Targets data.
"""

import pandas as pd


def create_gene_labels(opentargets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert gene-disease associations into binary labels.

    Args:
        opentargets_df: Filtered Open Targets data

    Returns:
        DataFrame with gene_id and label
    """

    disease_genes = set(opentargets_df["targetId"])

    labels_df = pd.DataFrame({
        "gene_id": list(disease_genes),
        "label": 1
    })

    return labels_df