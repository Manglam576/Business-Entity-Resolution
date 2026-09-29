# Business Entity Resolution Methodology

## 1. Normalization Strategy
The text features are aggressively normalized:
- Lowercasing to ensure case-insensitivity.
- Stripping legal suffixes like corp, llc, inc using regex boundaries to prevent matching merely on corporate structure.
- Removing non-alphanumeric punctuation while maintaining spacing.
- Expanding common street address abbreviations (e.g. rd -> road, st -> street).
- Extracting the first token of the normalized address as a "locality proxy".

## 2. Blocking / Candidate Generation
Generating an N x N cross-product is infeasible at scale, so we use a dual-blocking approach:
- **Vector Search (High Recall):** We generate text embeddings using `all-MiniLM-L6-v2` for a combined representation of `norm_name + norm_address + country`. A FAISS index (FlatIP for smaller scales or GPU Index if memory permits) is built over Source2 and Source3 embeddings. For each Source1 entity, we query the top 30 nearest neighbors.
- **Exact String Blocking (High Precision / Fallback):** As an auxiliary measure, candidates matching exactly on the first 4 characters of the normalized name and the country are also added (capped to prevent explosion).
This dual approach ensures a high recall ceiling in linear/sub-linear time.

## 3. Feature Engineering
For every candidate pair (S1 -> candidate), we compute:
- **Name Features:** Levenshtein ratio (using fast RapidFuzz implementation), token Jaccard similarity.
- **Address Features:** Levenshtein ratio, token Jaccard similarity.
- **Locality Match:** Binary flag indicating if the first alphanumeric address token matches.
- **Country Match:** Binary flag for exact country match.
- **Embedding Similarity:** The cosine similarity computed during the FAISS blocking step.
- **Name Length Bucket:** Binned string lengths.

## 4. Model Architecture
- **Classifier:** LightGBM Binary Classifier (LGBMClassifier).
- **Justification:** Boosted trees are incredibly fast to train and evaluate, natively handle missing values, and capture non-linear interactions well.
- **Class Imbalance:** Candidate generation produces heavily imbalanced datasets (few true matches among many retrieved candidates). We use `class_weight='balanced'` and inject all ground truth positive pairs into the training set.

## 5. Threshold Tuning and Validation
- A held-out validation set of 15% of the Source1 entities is evaluated.
- The pipeline scans probability thresholds from 0.3 to 0.95 in 0.05 increments.
- The exact macro-averaged F_0.5 score is computed at each threshold.
- The threshold that maximizes the validation F_0.5 is saved for inference.
