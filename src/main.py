"""
Main pipeline execution script.
"""

import os

from src.ingestion.load_string import load_string_data
from src.ingestion.load_opentargets import load_opentargets_data
from src.processing.create_labels import create_gene_labels
from src.config.paths import (
    RAW_STRING_PATH,
    RAW_OPENTARGETS_PATH,
    PROCESSED_INTERACTIONS,
    PROCESSED_LABELS,
)


def run_pipeline():
    """
    Run full data pipeline.
    """

    os.makedirs("data/processed", exist_ok=True)

    print("Loading STRING data...")
    string_df = load_string_data(RAW_STRING_PATH)
    print(f"STRING rows: {len(string_df)}")

    print("\nLoading Open Targets data...")
    opentargets_df = load_opentargets_data(RAW_OPENTARGETS_PATH)
    print(f"Open Targets rows: {len(opentargets_df)}")

    print("\nCreating labels...")
    labels_df = create_gene_labels(opentargets_df)
    print(f"Unique disease genes: {len(labels_df)}")

    # Save outputs
    string_df.to_csv(PROCESSED_INTERACTIONS, index=False)
    labels_df.to_csv(PROCESSED_LABELS, index=False)

    print("\nPipeline completed successfully.")
    print(f"Saved interactions → {PROCESSED_INTERACTIONS}")
    print(f"Saved labels → {PROCESSED_LABELS}")


if __name__ == "__main__":
    run_pipeline()