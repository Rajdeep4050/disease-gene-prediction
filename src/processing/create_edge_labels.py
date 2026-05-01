import pandas as pd


def create_edge_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Edge Label Definition:

    Possible cases after filtering:

    (1,0) → weak interaction → label = 0
    (0,1) → weak interaction → label = 0
    (1,1) → strong interaction → label = 1

    Removed earlier:
    (0,0) → useless

    Learning Objective:
    Distinguish strong disease interactions (1,1)
    from weak interactions (1,0) and (0,1)
    """

    df["edge_label"] = (
        (df["label_gene1"] == 1) &
        (df["label_gene2"] == 1)
    ).astype(int)

    return df