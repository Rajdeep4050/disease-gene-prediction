import pandas as pd


def map_string_to_gene(string_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert STRING protein interactions to gene interactions.
    """

    # Merge for protein1
    df = string_df.merge(
        mapping_df,
        left_on="protein1",
        right_on="protein_id",
        how="inner"
    ).rename(columns={"gene_id": "gene1"})

    df = df.drop(columns=["protein_id"])

    # Merge for protein2
    df = df.merge(
        mapping_df,
        left_on="protein2",
        right_on="protein_id",
        how="inner"
    ).rename(columns={"gene_id": "gene2"})

    df = df.drop(columns=["protein_id"])

    # Final clean dataset
    df = df[["gene1", "gene2", "combined_score"]]

    return df