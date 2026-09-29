import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import gc

def get_embeddings(texts, model_name='all-MiniLM-L6-v2', batch_size=256):
    import torch
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)
    # Generate embeddings
    embeddings = model.encode(texts.tolist(), batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings

def build_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    
    # Try GPU if available
    try:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        print("Using FAISS GPU index")
    except Exception as e:
        print("Using FAISS CPU index")
        
    index.add(embeddings)
    return index

def run_blocking(s1, s2, s3, k=20, out_path=None):
    # s2 and s3 are candidate sources
    s23 = pd.concat([s2, s3], ignore_index=True).reset_index(drop=True)
    
    # Compute embeddings
    print("Computing embeddings for Source 2+3...")
    s23_emb = get_embeddings(s23['text_for_embedding'])
    
    print("Computing embeddings for Source 1...")
    s1_emb = get_embeddings(s1['text_for_embedding'])
    
    # Build Index
    index = build_faiss_index(s23_emb)
    
    print("Searching nearest neighbors...")
    # Search top K
    distances, indices = index.search(s1_emb, k)
    
    # Build candidate pairs from FAISS
    pairs = []
    s1_ids = s1['entity_id'].values
    s23_ids = s23['entity_id'].values
    
    # For exact matching blocking key
    s1['blocking_key'] = s1['norm_name'].str[:4] + "_" + s1['country']
    s23['blocking_key'] = s23['norm_name'].str[:4] + "_" + s23['country']
    
    # Dictionary for exact blocks
    print("Building exact block index...")
    s23_block_dict = s23.groupby('blocking_key')['entity_id'].apply(list).to_dict()
    
    print("Assembling candidate pairs...")
    for i in range(len(s1_ids)):
        s1_id = s1_ids[i]
        
        # FAISS candidates
        for j in range(k):
            idx = indices[i, j]
            if idx >= 0 and idx < len(s23_ids):
                cand_id = s23_ids[idx]
                sim = distances[i, j]
                pairs.append((s1_id, cand_id, sim, 1)) # 1 indicates faiss
                
        # Exact block candidates (limit to avoid explosion)
        bkey = s1['blocking_key'].iloc[i]
        if bkey in s23_block_dict:
            exact_cands = s23_block_dict[bkey]
            # Add up to 10 random/first exact block candidates if not already covered
            # Note: deduplication will happen later
            for cand_id in exact_cands[:10]:
                pairs.append((s1_id, cand_id, 0.0, 0)) # 0 indicates exact block

    cand_df = pd.DataFrame(pairs, columns=['source1_entity_id', 'candidate_entity_id', 'embedding_sim', 'from_faiss'])
    # Deduplicate candidates per S1
    cand_df = cand_df.sort_values('embedding_sim', ascending=False).drop_duplicates(subset=['source1_entity_id', 'candidate_entity_id'], keep='first')
    
    if out_path:
        cand_df.to_csv(out_path, sep="\t", index=False)
        
    return cand_df, s23
