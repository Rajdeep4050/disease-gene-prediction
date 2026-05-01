import pandas as pd


def merge_graph_with_labels(gene_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach labels to graph nodes and filter meaningful edges.

    Cases BEFORE filtering:
    (0,0) → no disease genes → useless → removed
    (1,0) → weak interaction → kept
    (0,1) → weak interaction → kept
    (1,1) → strong interaction → kept

    After this step:
    Only edges with at least one disease gene remain.
    """

    # Label gene1
    df = gene_df.merge(
        labels_df,
        left_on="gene1",
        right_on="gene_id",
        how="left"
    ).rename(columns={"label": "label_gene1"})

    df = df.drop(columns=["gene_id"])

    # Label gene2
    df = df.merge(
        labels_df,
        left_on="gene2",
        right_on="gene_id",
        how="left"
    ).rename(columns={"label": "label_gene2"})

    df = df.drop(columns=["gene_id"])

    # Fill missing labels → non-disease = 0
    df["label_gene1"] = df["label_gene1"].fillna(0)
    df["label_gene2"] = df["label_gene2"].fillna(0)

    # 🔥 REMOVE useless edges (0,0)
    df = df[
        (df["label_gene1"] == 1) |
        (df["label_gene2"] == 1)
    ]

    df = df.reset_index(drop=True)

    return df