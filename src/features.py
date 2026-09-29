import pandas as pd
import numpy as np
from rapidfuzz import fuzz

def jaccard_similarity(s1, s2):
    set1 = set(str(s1).split())
    set2 = set(str(s2).split())
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def compute_features(pairs_df, s1_df, s23_df):
    """
    pairs_df: DataFrame with source1_entity_id, candidate_entity_id, embedding_sim
    s1_df: Normalized Source1 DataFrame
    s23_df: Normalized Source2+Source3 DataFrame
    """
    # Join with features
    s1_features = s1_df[['entity_id', 'norm_name', 'norm_address', 'country', 'address_first_token', 'name_len_bucket']].copy()
    s1_features.columns = ['source1_entity_id', 's1_name', 's1_address', 's1_country', 's1_addr_token', 's1_name_len']
    
    s23_features = s23_df[['entity_id', 'norm_name', 'norm_address', 'country', 'address_first_token', 'name_len_bucket']].copy()
    s23_features.columns = ['candidate_entity_id', 's23_name', 's23_address', 's23_country', 's23_addr_token', 's23_name_len']
    
    df = pairs_df.merge(s1_features, on='source1_entity_id', how='left')
    df = df.merge(s23_features, on='candidate_entity_id', how='left')
    
    # Feature 1: Name Levenshtein
    print("Computing name similarities...")
    df['name_levenshtein'] = df.apply(lambda x: fuzz.ratio(str(x['s1_name']), str(x['s23_name'])), axis=1)
    df['name_jaccard'] = df.apply(lambda x: jaccard_similarity(x['s1_name'], x['s23_name']), axis=1)
    
    # Feature 2: Address Levenshtein
    print("Computing address similarities...")
    df['address_levenshtein'] = df.apply(lambda x: fuzz.ratio(str(x['s1_address']), str(x['s23_address'])), axis=1)
    df['address_jaccard'] = df.apply(lambda x: jaccard_similarity(x['s1_address'], x['s23_address']), axis=1)
    
    # Feature 3: Locality token overlap
    df['locality_match'] = (df['s1_addr_token'] == df['s23_addr_token']).astype(int)
    # Mask if one is empty
    df.loc[(df['s1_addr_token'] == '') | (df['s23_addr_token'] == ''), 'locality_match'] = 0
    
    # Feature 4: Country match flag
    df['country_match'] = (df['s1_country'] == df['s23_country']).astype(int)
    df.loc[(df['s1_country'] == '') | (df['s23_country'] == ''), 'country_match'] = 0
    
    # We already have embedding_sim from pairs_df
    
    feature_cols = [
        'embedding_sim', 'name_levenshtein', 'name_jaccard', 
        'address_levenshtein', 'address_jaccard',
        'locality_match', 'country_match',
        's1_name_len', 's23_name_len'
    ]
    
    return df, feature_cols
