import argparse
import os
import time
import gc
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from src.features import compute_features
from src.normalization import extract_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--chunk_size", type=int, default=50000)
    args = parser.parse_args()

    start_time = time.time()
    
    # Load model
    model_path = os.path.join(args.model_dir, "fast_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    clf = model_data['model']
    feat_cols = model_data['features']
    best_threshold = model_data['threshold']

    print("--- MEMORY EFFICIENT FULL INFERENCE ---")
    usecols = ['entity_id', 'business_name', 'business_address', 'country']
    dtypes = {'entity_id': 'string[pyarrow]', 'business_name': 'string[pyarrow]', 'business_address': 'string[pyarrow]', 'country': 'string[pyarrow]'}

    print("Loading candidates into pyarrow strings (low memory)...")
    s2 = pd.read_csv(os.path.join(args.test_dir, 'test_source2.tsv'), sep='\t', usecols=usecols, dtype=dtypes)
    s3 = pd.read_csv(os.path.join(args.test_dir, 'test_source3.tsv'), sep='\t', usecols=usecols, dtype=dtypes)
    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()

    s23_norm = extract_features(s23)
    del s23
    gc.collect()

    print("Building exact string block dictionary...")
    s23_norm['blocking_key'] = s23_norm['norm_name'].str[:4] + "_" + s23_norm['country'].fillna('').astype(str)
    s23_block_dict = s23_norm.groupby('blocking_key')['entity_id'].apply(list).to_dict()

    print(f"Starting chunked inference on Source 1... (chunk size = {args.chunk_size})")
    reader = pd.read_csv(os.path.join(args.test_dir, 'test_source1.tsv'), sep='\t', chunksize=args.chunk_size, usecols=usecols, dtype=dtypes)

    cand_out = os.path.join(args.out_dir, "candidate_pairs.tsv")
    match_out = os.path.join(args.out_dir, "matching_results.tsv")

    for i, chunk in enumerate(reader):
        c_time = time.time()
        print(f"--- Processing Chunk {i+1} ({len(chunk)} rows) ---", flush=True)
        
        s1_norm = extract_features(chunk)
        s1_norm['blocking_key'] = s1_norm['norm_name'].str[:4] + "_" + s1_norm['country'].fillna('').astype(str)
        
        pairs = []
        for _, row in s1_norm.iterrows():
            bkey = row['blocking_key']
            cands = s23_block_dict.get(bkey, [])[:20]
            for c in cands:
                pairs.append({'source1_entity_id': row['entity_id'], 'candidate_entity_id': c, 'embedding_sim': 0.0})
                
        cand_df = pd.DataFrame(pairs)
        if cand_df.empty:
            cand_grouped = s1_norm[['entity_id']].copy()
            cand_grouped.columns = ['source1_entity_id']
            cand_grouped['candidate_entity_ids'] = ''
            
            match_grouped = cand_grouped.copy()
            match_grouped.columns = ['source1_entity_id', 'matched_entity_ids']
        else:
            cand_grouped = cand_df.groupby('source1_entity_id')['candidate_entity_id'].apply(lambda x: ','.join(x.astype(str))).reset_index()
            cand_grouped.columns = ['source1_entity_id', 'candidate_entity_ids']
            
            chunk_feat, _ = compute_features(cand_df, s1_norm, s23_norm)
            X_test = chunk_feat[feat_cols].astype(np.float32)
            chunk_feat['score'] = clf.predict_proba(X_test)[:, 1]
            matches = chunk_feat[chunk_feat['score'] >= best_threshold]
            
            match_grouped = matches.groupby('source1_entity_id')['candidate_entity_id'].apply(lambda x: ','.join(x.astype(str))).reset_index()
            match_grouped.columns = ['source1_entity_id', 'matched_entity_ids']
            
            all_s1 = s1_norm[['entity_id']].copy()
            all_s1.columns = ['source1_entity_id']
            cand_grouped = all_s1.merge(cand_grouped, on='source1_entity_id', how='left').fillna('')
            match_grouped = all_s1.merge(match_grouped, on='source1_entity_id', how='left').fillna('')
        
        mode = 'w' if i == 0 else 'a'
        header = True if i == 0 else False
        cand_grouped.to_csv(cand_out, sep='\t', index=False, mode=mode, header=header)
        match_grouped.to_csv(match_out, sep='\t', index=False, mode=mode, header=header)
        
        print(f"Chunk {i+1} done in {time.time()-c_time:.2f}s", flush=True)
        gc.collect()
        
    print(f"Total time: {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    main()
