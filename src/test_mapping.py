# import pandas as pd

# path = "data/raw/9606.protein.aliases.v12.0.txt"

# df = pd.read_csv(
#     path,
#     sep="\t",
#     compression="gzip"
# )

# # 🔥 Filter only ENSG mapping rows
# mapping_df = df[df["source"] == "Ensembl_HGNC_ensembl_gene_id"]

# print("Columns:\n", mapping_df.columns)
# print("\nSample:\n", mapping_df.head())
# print("\nShape:", mapping_df.shape)

# # testing processing/create_mapping.py

# from src.processing.create_mapping import create_protein_gene_mapping

# mapping = create_protein_gene_mapping(
#     "data/raw/9606.protein.aliases.v12.0.txt"
# )

# print(mapping.head())
# print("\nShape:", mapping.shape)



# # testing processing/map_string_to_gene.py
# from src.ingestion.load_string import load_string_data
# from src.processing.create_mapping import create_protein_gene_mapping
# from src.processing.map_string_to_gene import map_string_to_gene

# # Load STRING
# string_df = load_string_data("data/raw/9606.protein.links.v12.0.txt")

# # Load mapping
# mapping_df = create_protein_gene_mapping(
#     "data/raw/9606.protein.aliases.v12.0.txt"
# )

# # Apply mapping
# gene_df = map_string_to_gene(string_df, mapping_df)

# print(gene_df.head())
# print("\nShape:", gene_df.shape)




# # testing processing/merge_graph_labels.py
# from src.ingestion.load_string import load_string_data
# from src.processing.create_mapping import create_protein_gene_mapping
# from src.processing.map_string_to_gene import map_string_to_gene
# from src.processing.create_labels import create_gene_labels
# from src.ingestion.load_opentargets import load_opentargets_data
# from src.processing.merge_graph_labels import merge_graph_with_labels

# # Load STRING
# string_df = load_string_data("data/raw/9606.protein.links.v12.0.txt")

# # Load mapping
# mapping_df = create_protein_gene_mapping(
#     "data/raw/9606.protein.aliases.v12.0.txt"
# )

# # Convert to gene graph
# gene_df = map_string_to_gene(string_df, mapping_df)

# # Load labels
# ot_df = load_opentargets_data("data/raw/opentargets/")
# labels_df = create_gene_labels(ot_df)

# # Merge
# final_df = merge_graph_with_labels(gene_df, labels_df)

# print(final_df.head())
# print("\nShape:", final_df.shape)



# # testing processing/create_edge_labels.py
# from src.ingestion.load_string import load_string_data
# from src.processing.create_mapping import create_protein_gene_mapping
# from src.processing.map_string_to_gene import map_string_to_gene
# from src.processing.create_labels import create_gene_labels
# from src.ingestion.load_opentargets import load_opentargets_data
# from src.processing.merge_graph_labels import merge_graph_with_labels
# from src.processing.create_edge_labels import create_edge_labels
# from src.processing.create_features import add_node_degree_features

# # 1. Load STRING
# string_df = load_string_data("data/raw/9606.protein.links.v12.0.txt")

# # 2. Load mapping
# mapping_df = create_protein_gene_mapping(
#     "data/raw/9606.protein.aliases.v12.0.txt"
# )

# # 3. Convert to gene graph
# gene_df = map_string_to_gene(string_df, mapping_df)

# # 4. Load Open Targets
# ot_df = load_opentargets_data("data/raw/opentargets/")

# # 5. Create gene labels
# labels_df = create_gene_labels(ot_df)

# # 6. Merge graph + labels
# merged_df = merge_graph_with_labels(gene_df, labels_df)

# # 7. Create edge labels (THIS was failing)
# final_df = create_edge_labels(merged_df)

# # Output
# print(final_df.head())

# print("\nShape:", final_df.shape)

# print("\n===== CASE BREAKDOWN =====")

# case_counts = (
#     final_df.groupby(["label_gene1", "label_gene2"])
#     .size()
#     .reset_index(name="count")
# )

# print(case_counts)

# print("\n===== EDGE LABEL DISTRIBUTION =====")
# print(final_df["edge_label"].value_counts())

# print("\n===== POSITIVE RATIO =====")
# print(final_df["edge_label"].mean())





# testing processing/create_features.py
from src.processing.create_features import (
    add_neighbor_disease_count,
    add_neighbor_disease_ratio
)
from src.ingestion.load_string import load_string_data
from src.processing.create_mapping import create_protein_gene_mapping
from src.processing.map_string_to_gene import map_string_to_gene
from src.processing.create_labels import create_gene_labels
from src.ingestion.load_opentargets import load_opentargets_data
from src.processing.merge_graph_labels import merge_graph_with_labels
from src.processing.create_edge_labels import create_edge_labels
from src.processing.create_features import add_node_degree_features

# 1. Load STRING
string_df = load_string_data("data/raw/9606.protein.links.v12.0.txt")

# 2. Load mapping
mapping_df = create_protein_gene_mapping(
    "data/raw/9606.protein.aliases.v12.0.txt"
)

# 3. Convert to gene graph
gene_df = map_string_to_gene(string_df, mapping_df)

# 4. Load Open Targets
ot_df = load_opentargets_data("data/raw/opentargets/")

# 5. Create gene labels
labels_df = create_gene_labels(ot_df)

# 6. Merge graph + labels
merged_df = merge_graph_with_labels(gene_df, labels_df)

# 7. Create edge labels (THIS was failing)
final_df = create_edge_labels(merged_df)

# Output
print(final_df.head())

print("\nShape:", final_df.shape)

print("\n===== CASE BREAKDOWN =====")

case_counts = (
    final_df.groupby(["label_gene1", "label_gene2"])
    .size()
    .reset_index(name="count")
)

print(case_counts)

print("\n===== EDGE LABEL DISTRIBUTION =====")
print(final_df["edge_label"].value_counts())

print("\n===== POSITIVE RATIO =====")
print(final_df["edge_label"].mean())

# Apply features
final_df = add_node_degree_features(final_df)
final_df = add_neighbor_disease_count(final_df)
final_df = add_neighbor_disease_ratio(final_df)

print("\n===== FEATURE SAMPLE =====")
print(final_df[[
    "gene1",
    "gene2",
    "degree_gene1",
    "degree_gene2",
    "disease_neighbors_gene1",
    "disease_neighbors_gene2",
    "disease_ratio_gene1",
    "disease_ratio_gene2"
]].head())

final_df = add_node_degree_features(final_df)

print(final_df.head())