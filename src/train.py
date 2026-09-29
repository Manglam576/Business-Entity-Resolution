import argparse
import os
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from src.features import compute_features
from src.normalization import extract_features

def compute_f05_score(ground_truth, predictions):
    tp = 0
    fp = 0
    fn = 0
    
    all_queries = set(ground_truth.keys()).union(set(predictions.keys()))
    for q in all_queries:
        gt_set = ground_truth.get(q, set())
        pred_set = predictions.get(q, set())
        
        tp += len(gt_set.intersection(pred_set))
        fp += len(pred_set - gt_set)
        fn += len(gt_set - pred_set)
        
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return f05

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    print("--- TRAINING FAST MODEL ---")
    
    # Load ground truth
    gt_path = os.path.join(args.train_dir, 'train_ground_truth.tsv')
    gt = pd.read_csv(gt_path, sep='\t')
    gt_dict = {}
    for _, row in gt.iterrows():
        if not pd.isna(row['matched_entity_ids']):
            gt_dict[row['source1_entity_id']] = set(str(row['matched_entity_ids']).split(','))

    print("Loading training data...")
    s1_train = pd.read_csv(os.path.join(args.train_dir, 'train_source1.tsv'), sep='\t')
    s2_train = pd.read_csv(os.path.join(args.train_dir, 'train_source2.tsv'), sep='\t')
    s3_train = pd.read_csv(os.path.join(args.train_dir, 'train_source3.tsv'), sep='\t')

    s1_train_norm = extract_features(s1_train)
    s23_train_norm = extract_features(pd.concat([s2_train, s3_train], ignore_index=True))

    print("Building exact block dict for training...")
    s1_train_norm['blocking_key'] = s1_train_norm['norm_name'].str[:4] + "_" + s1_train_norm['country'].fillna('').astype(str)
    s23_train_norm['blocking_key'] = s23_train_norm['norm_name'].str[:4] + "_" + s23_train_norm['country'].fillna('').astype(str)

    block_dict = s23_train_norm.groupby('blocking_key')['entity_id'].apply(list).to_dict()

    print("Assembling positive and negative pairs...")
    pairs = []
    for _, row in s1_train_norm.iterrows():
        s1_id = row['entity_id']
        bkey = row['blocking_key']
        
        # Add some exact block candidates (mostly negatives, some positives)
        cands = block_dict.get(bkey, [])[:15]
        for c in cands:
            label = 1 if c in gt_dict.get(s1_id, set()) else 0
            pairs.append({'source1_entity_id': s1_id, 'candidate_entity_id': c, 'embedding_sim': 0.0, 'from_faiss': 0, 'label': label})
            
        # Inject ground truth to ensure model learns positives
        for c in gt_dict.get(s1_id, set()):
            pairs.append({'source1_entity_id': s1_id, 'candidate_entity_id': c, 'embedding_sim': 0.0, 'from_faiss': 0, 'label': 1})
            
    train_df = pd.DataFrame(pairs).drop_duplicates(subset=['source1_entity_id', 'candidate_entity_id'])
    
    print("Computing string features...")
    train_feat, feat_cols = compute_features(train_df, s1_train_norm, s23_train_norm)
    feat_cols.remove('embedding_sim') # Drop embedding_sim for the fast memory-efficient model

    print("Training LightGBM classifier...")
    X = train_feat[feat_cols].astype(np.float32)
    y = train_feat['label']
    clf = lgb.LGBMClassifier(n_estimators=100, num_leaves=31, learning_rate=0.05, n_jobs=1)
    clf.fit(X, y)

    # Simplified threshold for fast pipeline
    best_threshold = 0.5
    
    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, 'fast_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({'model': clf, 'features': feat_cols, 'threshold': best_threshold}, f)
        
    print(f"Model saved to {model_path} with threshold {best_threshold}")

if __name__ == "__main__":
    main()
