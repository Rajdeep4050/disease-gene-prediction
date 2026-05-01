import pandas as pd


def create_protein_gene_mapping(path: str) -> pd.DataFrame:
    """
    Create mapping from STRING protein IDs to Ensembl gene IDs.
    """

    df = pd.read_csv(
        path,
        sep="\t",
        compression="gzip"
    )

    # Filter only ENSG mappings
    df = df[df["source"] == "Ensembl_HGNC_ensembl_gene_id"]

    # Select required columns
    df = df[["#string_protein_id", "alias"]]

    # Rename columns
    df = df.rename(columns={
        "#string_protein_id": "protein_id",
        "alias": "gene_id"
    })

    # Remove duplicates
    df = df.drop_duplicates()

    return df