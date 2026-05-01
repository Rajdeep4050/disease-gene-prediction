import pandas as pd

from src.ingestion.load_string import load_string_data
from src.processing.create_mapping import create_protein_gene_mapping
from src.processing.map_string_to_gene import map_string_to_gene
from src.processing.create_labels import create_gene_labels
from src.ingestion.load_opentargets import load_opentargets_data

from src.processing.merge_graph_labels import merge_graph_with_labels
from src.processing.create_edge_labels import create_edge_labels
from src.processing.create_features import (
    add_node_degree_features,
    add_common_neighbors,
    add_jaccard_similarity
)


def main():
    print("🚀 Building FINAL dataset (correct pipeline)...\n")

    # =========================
    # STEP 1: LOAD STRING (protein level)
    # =========================
    string_df = load_string_data("data/raw/9606.protein.links.v12.0.txt")
    print(f"STRING: {string_df.shape}")

    # =========================
    # STEP 2: LOAD MAPPING (ENSP → ENSG)
    # =========================
    mapping_df = create_protein_gene_mapping(
        "data/raw/9606.protein.aliases.v12.0.txt"
    )
    print(f"Mapping: {mapping_df.shape}")

    # =========================
    # STEP 3: CONVERT TO GENE GRAPH
    # =========================
    gene_df = map_string_to_gene(string_df, mapping_df)
    print(f"Gene graph: {gene_df.shape}")

    # =========================
    # STEP 4: LOAD OPEN TARGETS
    # =========================
    ot_df = load_opentargets_data("data/raw/opentargets/")
    print(f"Open Targets: {ot_df.shape}")

    # =========================
    # STEP 5: CREATE LABELS
    # =========================
    labels_df = create_gene_labels(ot_df)
    print(f"Labels: {labels_df.shape}")

    # =========================
    # STEP 6: MERGE GRAPH + LABELS
    # =========================
    merged_df = merge_graph_with_labels(gene_df, labels_df)
    print(f"Merged: {merged_df.shape}")

    # =========================
    # STEP 7: EDGE LABEL
    # =========================
    final_df = create_edge_labels(merged_df)

    print("Label distribution:")
    print(final_df["edge_label"].value_counts())

    # =========================
    # STEP 8: FEATURES
    # =========================
    final_df = add_node_degree_features(final_df)
    final_df = add_common_neighbors(final_df)
    final_df = add_jaccard_similarity(final_df)

    print(f"Final dataset: {final_df.shape}")

    # =========================
    # STEP 9: SAVE
    # =========================
    final_df.to_csv("data/processed/final_dataset.csv", index=False)

    print("\n✅ Saved: data/processed/final_dataset.csv")


if __name__ == "__main__":
    main()