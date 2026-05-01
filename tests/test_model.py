# # from src.ingestion.load_string import load_string_data
# # from src.processing.create_mapping import create_protein_gene_mapping
# # from src.processing.map_string_to_gene import map_string_to_gene
# # from src.ingestion.load_opentargets import load_opentargets_data
# # from src.processing.create_labels import create_gene_labels
# # from src.processing.merge_graph_labels import merge_graph_with_labels
# # from src.processing.create_edge_labels import create_edge_labels
# # from src.processing.create_features import (
# #     add_node_degree_features,
# #     add_neighbor_disease_count,
# #     add_neighbor_disease_ratio
# # )
# # from src.model.train_model import train_models
# # from src.processing.create_features import add_common_neighbors


# # def run_pipeline_and_train():
# #     print("===== LOADING DATA =====")

# #     # Load STRING
# #     string_df = load_string_data("data/raw/9606.protein.links.v12.0.txt")

# #     # Load mapping
# #     mapping_df = create_protein_gene_mapping(
# #         "data/raw/9606.protein.aliases.v12.0.txt"
# #     )

# #     # Convert to gene graph
# #     gene_df = map_string_to_gene(string_df, mapping_df)

# #     # Load Open Targets
# #     ot_df = load_opentargets_data("data/raw/opentargets/")

# #     # Create labels
# #     labels_df = create_gene_labels(ot_df)

# #     # Merge graph + labels
# #     merged_df = merge_graph_with_labels(gene_df, labels_df)

# #     # Create edge labels
# #     final_df = create_edge_labels(merged_df)

# #     print("\n===== FEATURE ENGINEERING =====")

# #     final_df = add_node_degree_features(final_df)
# #     final_df = add_neighbor_disease_count(final_df)
# #     final_df = add_neighbor_disease_ratio(final_df)

# #     print("\nDataset shape:", final_df.shape)

# #     print("\n===== TRAINING MODELS =====")

# #     train_models(final_df)


# # if __name__ == "__main__":
# #     run_pipeline_and_train()
    


# from src.ingestion.load_string import load_string_data
# from src.processing.create_mapping import create_protein_gene_mapping
# from src.processing.map_string_to_gene import map_string_to_gene
# from src.ingestion.load_opentargets import load_opentargets_data
# from src.processing.create_labels import create_gene_labels
# from src.processing.merge_graph_labels import merge_graph_with_labels
# from src.processing.create_edge_labels import create_edge_labels
# from src.processing.create_features import (
#     add_node_degree_features,
#     add_common_neighbors
# )
# from src.model.train_model import train_models


# def run_pipeline_and_train():
#     print("===== LOADING DATA =====")

#     # Load STRING data
#     string_df = load_string_data("data/raw/9606.protein.links.v12.0.txt")

#     # Load mapping (protein → gene)
#     mapping_df = create_protein_gene_mapping(
#         "data/raw/9606.protein.aliases.v12.0.txt"
#     )

#     # Convert protein graph → gene graph
#     gene_df = map_string_to_gene(string_df, mapping_df)

#     # Load Open Targets data
#     ot_df = load_opentargets_data("data/raw/opentargets/")

#     # Create gene labels
#     labels_df = create_gene_labels(ot_df)

#     # Merge graph with labels
#     merged_df = merge_graph_with_labels(gene_df, labels_df)

#     # Create edge labels
#     final_df = create_edge_labels(merged_df)

#     print("\n===== FEATURE ENGINEERING =====")

#     # Add features (SAFE — no leakage)
#     final_df = add_node_degree_features(final_df)
#     final_df = add_common_neighbors(final_df)

#     print("\nDataset shape:", final_df.shape)

#     print("\n===== TRAINING MODELS =====")

#     # Train models
#     train_models(final_df)


# if __name__ == "__main__":
#     run_pipeline_and_train()





# from src.ingestion.load_string import load_string_data
# from src.processing.create_mapping import create_protein_gene_mapping
# from src.processing.map_string_to_gene import map_string_to_gene
# from src.ingestion.load_opentargets import load_opentargets_data
# from src.processing.create_labels import create_gene_labels
# from src.processing.merge_graph_labels import merge_graph_with_labels
# from src.processing.create_edge_labels import create_edge_labels
# from src.processing.create_features import (
#     add_node_degree_features,
#     add_neighbor_disease_count,
#     add_neighbor_disease_ratio
# )
# from src.model.train_model import train_models
# from src.processing.create_features import add_common_neighbors


# def run_pipeline_and_train():
#     print("===== LOADING DATA =====")

#     # Load STRING
#     string_df = load_string_data("data/raw/9606.protein.links.v12.0.txt")

#     # Load mapping
#     mapping_df = create_protein_gene_mapping(
#         "data/raw/9606.protein.aliases.v12.0.txt"
#     )

#     # Convert to gene graph
#     gene_df = map_string_to_gene(string_df, mapping_df)

#     # Load Open Targets
#     ot_df = load_opentargets_data("data/raw/opentargets/")

#     # Create labels
#     labels_df = create_gene_labels(ot_df)

#     # Merge graph + labels
#     merged_df = merge_graph_with_labels(gene_df, labels_df)

#     # Create edge labels
#     final_df = create_edge_labels(merged_df)

#     print("\n===== FEATURE ENGINEERING =====")

#     final_df = add_node_degree_features(final_df)
#     final_df = add_neighbor_disease_count(final_df)
#     final_df = add_neighbor_disease_ratio(final_df)

#     print("\nDataset shape:", final_df.shape)

#     print("\n===== TRAINING MODELS =====")

#     train_models(final_df)


# if __name__ == "__main__":
#     run_pipeline_and_train()
    


from src.ingestion.load_string import load_string_data
from src.processing.create_mapping import create_protein_gene_mapping
from src.processing.map_string_to_gene import map_string_to_gene
from src.ingestion.load_opentargets import load_opentargets_data
from src.processing.create_labels import create_gene_labels
from src.processing.merge_graph_labels import merge_graph_with_labels
from src.processing.create_edge_labels import create_edge_labels
from src.processing.create_features import (
    add_node_degree_features,
    add_common_neighbors,
    add_jaccard_similarity
)
from src.model.train_model import train_models


def run_pipeline_and_train():
    print("===== LOADING DATA =====")

    # Load STRING data
    string_df = load_string_data("data/raw/9606.protein.links.v12.0.txt")

    # Load mapping (protein → gene)
    mapping_df = create_protein_gene_mapping(
        "data/raw/9606.protein.aliases.v12.0.txt"
    )

    # Convert protein graph → gene graph
    gene_df = map_string_to_gene(string_df, mapping_df)

    # Load Open Targets data
    ot_df = load_opentargets_data("data/raw/opentargets/")

    # Create gene labels
    labels_df = create_gene_labels(ot_df)

    # Merge graph with labels
    merged_df = merge_graph_with_labels(gene_df, labels_df)

    # Create edge labels
    final_df = create_edge_labels(merged_df)

    print("\n===== FEATURE ENGINEERING =====")

    # Add features (SAFE — no leakage)
    final_df = add_node_degree_features(final_df)
    final_df = add_common_neighbors(final_df)
    final_df = add_jaccard_similarity(final_df)
    print("\nDataset shape:", final_df.shape)

    print("\n===== TRAINING MODELS =====")

    # Train models
    train_models(final_df)


if __name__ == "__main__":
    run_pipeline_and_train()