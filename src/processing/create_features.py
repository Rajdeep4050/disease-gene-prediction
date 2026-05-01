import pandas as pd


def add_node_degree_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 1: Node Degree

    Measures how many interactions a gene has.

    Interpretation:
    - High degree → highly connected gene (hub)
    - Low degree → less connected gene
    """

    all_nodes = pd.concat([df["gene1"], df["gene2"]])

    degree = all_nodes.value_counts().to_dict()

    df["degree_gene1"] = df["gene1"].map(degree)
    df["degree_gene2"] = df["gene2"].map(degree)

    return df


def add_neighbor_disease_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 2: Disease Neighbor Count

    Counts how many neighbors of a gene are disease-related.

    Why this matters:
    - Not all connections are equal
    - A gene connected to many disease genes is more important

    Example:
    Gene A → 100 connections, 2 disease → weak
    Gene B → 50 connections, 30 disease → strong
    """

    # Reverse edges to make graph undirected
    df_rev = df.rename(columns={
        "gene1": "gene2",
        "gene2": "gene1",
        "label_gene1": "label_gene2",
        "label_gene2": "label_gene1"
    })

    full_df = pd.concat([df, df_rev])

    # Count disease neighbors
    disease_neighbors = (
        full_df[full_df["label_gene2"] == 1]
        .groupby("gene1")
        .size()
        .to_dict()
    )

    df["disease_neighbors_gene1"] = df["gene1"].map(disease_neighbors).fillna(0)
    df["disease_neighbors_gene2"] = df["gene2"].map(disease_neighbors).fillna(0)

    return df


def add_neighbor_disease_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 3: Disease Neighbor Ratio

    Ratio = disease_neighbors / total_degree

    Why this matters:
    - Normalizes for hub bias
    - Prevents high-degree nodes from dominating

    Example:
    Gene A → 100 degree, 10 disease → 0.1
    Gene B → 20 degree, 10 disease → 0.5 (more important)
    """

    df["disease_ratio_gene1"] = (
        df["disease_neighbors_gene1"] / df["degree_gene1"]
    ).fillna(0)

    df["disease_ratio_gene2"] = (
        df["disease_neighbors_gene2"] / df["degree_gene2"]
    ).fillna(0)

    return df

def add_common_neighbors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count number of common neighbors between gene1 and gene2.
    """

    from collections import defaultdict

    neighbors = defaultdict(set)

    for _, row in df.iterrows():
        neighbors[row["gene1"]].add(row["gene2"])
        neighbors[row["gene2"]].add(row["gene1"])

    common_counts = []

    for _, row in df.iterrows():
        g1 = row["gene1"]
        g2 = row["gene2"]

        common = neighbors[g1].intersection(neighbors[g2])
        common_counts.append(len(common))

    df["common_neighbors"] = common_counts

    return df

def add_jaccard_similarity(df):
    """
    Jaccard similarity between gene1 and gene2
    """

    from collections import defaultdict

    neighbors = defaultdict(set)

    for _, row in df.iterrows():
        neighbors[row["gene1"]].add(row["gene2"])
        neighbors[row["gene2"]].add(row["gene1"])

    jaccard_scores = []

    for _, row in df.iterrows():
        g1 = row["gene1"]
        g2 = row["gene2"]

        n1 = neighbors[g1]
        n2 = neighbors[g2]

        intersection = len(n1 & n2)
        union = len(n1 | n2)

        score = intersection / union if union != 0 else 0
        jaccard_scores.append(score)

    df["jaccard_similarity"] = jaccard_scores

    return df