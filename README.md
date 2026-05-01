# Disease-Gene Prediction

A modular pipeline that builds a labeled disease-gene interaction dataset from STRING and Open Targets data.

## Project overview

This repository prepares a graph-style dataset for disease gene prediction. It loads raw biological data, aligns protein and gene identifiers, constructs gene interaction edges, attaches disease labels, and produces labeled edges for downstream modeling.

Key capabilities:

- Load STRING protein-protein interactions
- Map STRING proteins to Ensembl gene IDs
- Load sampled Open Targets gene-disease association data
- Generate binary disease labels at the gene level
- Merge labels with graph edges
- Create edge-level labels for model training
- Train baseline models using graph-derived features

## Actual repository structure

```
disease-gene-prediction/
├── data/
│   ├── raw/
│   │   ├── 9606.protein.aliases.v12.0.txt
│   │   ├── 9606.protein.links.v12.0.txt
│   │   ├── opentargets/              # sampled Open Targets parquet files
│   │   └── __pycache__/
│   ├── processed/
│   └── download_opentargets.py
├── models/
│   ├── feature_importance.py
│   ├── save_model.py
│   ├── train_model.py
│   └── __pycache__/
├── requirements.txt
├── src/
│   ├── config/
│   │   ├── constants.py
│   │   └── paths.py
│   ├── ingestion/
│   │   ├── load_opentargets.py
│   │   └── load_string.py
│   ├── model/
│   │   ├── feature_importance.py
│   │   ├── save_model.py
│   │   └── train_model.py
│   ├── processing/
│   │   ├── create_edge_labels.py
│   │   ├── create_features.py
│   │   ├── create_labels.py
│   │   ├── create_mapping.py
│   │   ├── map_string_to_gene.py
│   │   └── merge_graph_labels.py
│   ├── __init__.py
│   └── main.py
├── tests/
│   ├── test_load.py
│   ├── test_mapping.py
│   ├── test_model.py
│   └── __init__.py
├── README.md
├── Reference.md
├── .gitignore
└── venv/
```

## Data sources

### STRING

https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz
https://stringdb-downloads.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz

https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/26.03/

- Protein-protein interactions for Homo sapiens
- Expected raw file: `data/raw/9606.protein.links.v12.0.txt`
- Filtered by `combined_score` using `src.config.constants.STRING_SCORE_THRESHOLD`

### Open Targets

- Gene-disease association dataset stored as parquet files
- Raw folder: `data/raw/opentargets/`
- Sampled using `src.ingestion.load_opentargets.load_opentargets_data`
- Filtered by `associationScore` using `src.config.constants.OPENTARGETS_SCORE_THRESHOLD`

## Core pipeline

### 1. Load raw data

- STRING: `src.ingestion.load_string.load_string_data`
- Open Targets: `src.ingestion.load_opentargets.load_opentargets_data`

### 2. Create gene labels

- `src.processing.create_labels.create_gene_labels`
- Converts Open Targets `targetId` values into disease gene labels
- Produces `gene_id` / `label` pairs where `label = 1`

### 3. Map STRING proteins to genes

- `src.processing.create_mapping.create_protein_gene_mapping`
- Reads `data/raw/9606.protein.aliases.v12.0.txt`
- Keeps only Ensembl gene mappings
- Produces a `protein_id -> gene_id` table

### 4. Convert protein edges to gene edges

- `src.processing.map_string_to_gene.map_string_to_gene`
- Joins protein interaction rows to gene IDs for both endpoints
- Outputs `gene1`, `gene2`, `combined_score`

### 5. Merge graph and labels

- `src.processing.merge_graph_labels.merge_graph_with_labels`
- Attaches `label_gene1` and `label_gene2` for each edge
- Fills missing gene labels as `0`
- Removes edges where both genes are non-disease

### 6. Create edge labels

- `src.processing.create_edge_labels.create_edge_labels`
- Assigns `edge_label = 1` when both genes are disease-associated
- Assigns `edge_label = 0` when exactly one gene is disease-associated

## Feature engineering

The repository also includes graph-based feature generation in `src.processing.create_features.py`:

- node degree for each gene endpoint
- disease neighbor count and ratio
- common neighbors
- Jaccard similarity between gene neighborhoods

## Modeling support

Training utilities live under `src.model/`:

- `train_model.py` trains Logistic Regression and Random Forest models
- `feature_importance.py` reports feature importance
- `save_model.py` saves a trained model

## Outputs

The main pipeline saves processed artifacts to:

- `data/processed/interactions.csv`
- `data/processed/gene_labels.csv`

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main data pipeline:

```bash
python -m src.main
```

Train models manually from the model script if needed:

```bash
python -m src.model.train_model
```

## Testing

Run tests with:

```bash
python -m pytest
```

or if you prefer direct module execution:

```bash
python -m tests.test_mapping
python -m tests.test_load
python -m tests.test_model
```

## Notes

- The current pipeline keeps only edges with at least one disease-related gene.
- `edge_label = 1` denotes a strong disease-disease interaction.
- `edge_label = 0` denotes a mixed edge with one disease gene and one non-disease gene.

## Dependencies

- pandas
- numpy
- scikit-learn
- pyarrow
- requests
- beautifulsoup4

## Suggested next steps

- add integration tests for the full pipeline
- extend label coverage beyond Open Targets sampling
- build a dedicated model evaluation script
- add visualization notebooks for network analysis
