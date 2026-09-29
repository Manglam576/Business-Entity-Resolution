# Business Entity Resolution Pipeline

## Overview
This repository contains my end-to-end Machine Learning pipeline for resolving business identities across multiple noisy datasets. 

In large-scale commercial platforms, business identity data arrives from multiple independent sources — each contributing partial, noisy fragments of information about the same real-world entities. These fragments share no common identifiers. This project solves the **Entity Resolution (ER)** challenge by matching business records from 3 independent data sources with noisy and inconsistent fields (such as variations in legal suffixes, missing address components, and typos).

## Repository Structure
- `src/` : Contains the core Python pipeline, including blocking, feature engineering, training, and inference.
- `utils/` : Helper scripts for validating output formats and creating data samples.
- `Documentation_template.md` : Detailed methodology, candidate generation strategy, and model architecture.
- `requirements.txt` : Python dependencies required to run the pipeline.

## Methodology
The pipeline takes a two-step approach:
1. **Candidate Generation (Blocking):** Efficiently narrows down millions of possible pairs to a manageable subset by matching on geographic and name prefixes.
2. **Machine Learning Matching:** Uses LightGBM on engineered string-similarity features (Levenshtein, Jaccard, Token Match) to score pairs and generate final entity matches.

*See `Documentation_template.md` for a comprehensive write-up of the methodology and architecture.*

## Available Models (Brains)
This repository includes two trained models in the `output/` directory, depending on your computational constraints:
- **`fast_model.pkl`**: This is the default production model. It uses 8 string-similarity features. It drops heavy semantic embedding features to ensure inference runs extremely fast and memory-efficiently across millions of candidates.
- **`lgb_model.pkl`**: This is the highest-accuracy model. It uses 9 features, including heavy semantic embeddings (`embedding_sim`). Use this if you have high memory/compute availability and need maximum precision.
## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Training the Model
To train the LightGBM model on the ground-truth dataset:
```bash
python src/train.py \
  --train_dir dataset/train \
  --out_dir output
```

### 3. Running Inference
To generate predictions (candidate pairs and final matches) on the test set:
```bash
python src/infer.py \
  --test_dir dataset/test \
  --model_dir output \
  --out_dir output
```

### 4. Validating Output
To ensure the generated `matching_results.tsv` conforms to the format rules:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
