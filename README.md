# Disease-Gene Prediction

A machine learning project for predicting disease-gene associations.

## Project Structure

```
disease-gene-prediction/
│
├── data/
│   ├── raw/                      # Raw input data
│   ├── processed/                # Processed and cleaned data
│   └── download_opentargets.py  # Script to download data sources
│
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── load_string.py        # Load STRING interaction data
│   │   └── load_disgenet.py      # Load DisGeNET disease-gene associations
│   │
│   ├── processing/
│   │   ├── __init__.py
│   │   └── create_labels.py      # Create labels from disease-gene associations
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── paths.py              # Centralized path configuration
│   │   └── constants.py          # Project constants and thresholds
│   │
│   ├── __init__.py
│   └── main.py                   # Main pipeline orchestration
│
├── tests/                        # Unit tests for project modules
├── notebooks/                    # Jupyter notebooks for exploration
├── README.md
├── requirements.txt
└── .gitignore
```

## Problem

Predict disease-associated genes using biological interaction networks and sequence-based features.

## Data Sources

- STRING (protein interactions)
- DisGeNET (disease-gene associations)
- NCBI (gene sequences)

## Approach

Combine graph-based relationships with sequence-derived features to identify potential disease genes.

# Disease Gene Prediction using Graph-Based Learning

## Overview

This project aims to build a machine learning system that predicts whether a gene is associated with human diseases. The system integrates biological interaction data with engineered features derived from gene sequences and applies graph-based machine learning models to identify potential disease-related genes.

The goal is to assist in narrowing down candidate genes for further biological validation, reducing the need for exhaustive laboratory experiments.

---

## Problem Statement

Given a large set of genes and their interactions, along with known disease-associated genes, we aim to:

- Identify patterns in gene interaction networks
- Leverage simple sequence-derived features
- Predict which previously unlabeled genes are likely to be disease-associated

This is formulated as a **binary classification problem at the gene level**.

---

## Key Idea

The hypothesis behind this project is:

- Genes that interact with disease-associated genes are more likely to be disease-associated
- Genes with similar sequence characteristics to known disease genes may also share functional similarities

By combining:

1. **Graph structure (gene-gene interactions)**
2. **Sequence-derived features (basic biological signals)**

we can improve prediction performance compared to using either source alone.

---

## Data Sources

This project integrates multiple publicly available biological datasets:

### 1. Protein Interaction Data

Source: STRING database

- Contains protein-protein interactions
- Used to construct a graph where:
  - Nodes = genes/proteins
  - Edges = interactions with confidence scores

### 2. Disease-Gene Associations

Source: DisGeNET

- Contains known associations between genes and diseases
- Used to label genes as:
  - 1 → disease-associated
  - 0 → not known to be associated

### 3. Gene Sequence Data

Source: NCBI

- Contains DNA/protein sequences
- Used to derive features such as:
  - GC content
  - Sequence length
  - k-mer frequencies

---

## Data Pipeline

The data processing pipeline follows these steps:

1. Load raw datasets from different sources
2. Filter based on confidence thresholds:
   - STRING interaction score threshold
   - DisGeNET association score threshold

3. Normalize and align gene identifiers across datasets
4. Construct:
   - Interaction dataset (edges)
   - Gene label dataset

5. Generate feature vectors for each gene
6. Save processed datasets for modeling

---

## Feature Engineering

Instead of using heavy biological models, we extract simple and interpretable features:

- Sequence length
- GC content (proportion of G and C nucleotides)
- k-mer frequencies (subsequence patterns)

These features act as lightweight biological signals.

---

## Modeling Approach

The project includes multiple models for comparison:

### Baseline Models

- XGBoost / Logistic Regression using only sequence features

### Graph-Based Models

- Node2Vec (embedding-based approach)
- Graph Neural Network (GraphSAGE)

The final model uses:

- Graph structure (interaction network)
- Node features (sequence-derived features)

---

## Evaluation Strategy

Models are evaluated using:

- ROC-AUC
- Precision / Recall
- Comparison across:
  - Feature-only models
  - Graph-only models
  - Combined models

---

## Project Structure

The project follows a modular pipeline:

- ingestion: loading raw datasets
- processing: cleaning and label creation
- config: centralized constants and paths
- main pipeline orchestration

This structure ensures reproducibility and scalability.

---

## Engineering Principles

- Modular code design
- Separation of concerns (data, processing, modeling)
- Config-driven thresholds and paths
- Reproducible pipelines
- Version control using GitHub

---

## Expected Outcome

The system should:

- Identify candidate disease genes
- Demonstrate improvement using combined graph + feature approach
- Provide a reproducible pipeline for future experimentation

---

## Future Extensions

- Add explainability (feature importance, attention)
- Improve feature engineering
- Incorporate additional biological datasets
- Extend to link prediction (gene-disease pairs)

---

## Summary (Simple Explanation)

This project builds a system that:

- Looks at how genes interact
- Analyzes simple patterns in their sequences
- Predicts which genes are likely linked to diseases

The goal is to assist biological research with data-driven insights.
