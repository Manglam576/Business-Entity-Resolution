import pandas as pd
import re

def normalize_text_series(series):
    # Lowercase
    s = series.astype(str).str.lower()
    # Remove punctuation except spaces
    s = s.str.replace(r'[^\w\s]', ' ', regex=True)
    # Strip legal suffixes
    suffixes = r'\b(corp|corporation|ltd|limited|pvt|private|llc|inc|incorporated|co|company)\b'
    s = s.str.replace(suffixes, '', regex=True)
    # Expand abbreviations
    abbrevs = {
        r'\brd\b': 'road',
        r'\bst\b': 'street',
        r'\bave\b': 'avenue',
        r'\bblvd\b': 'boulevard',
        r'\bdr\b': 'drive',
        r'\bln\b': 'lane',
        r'\bct\b': 'court',
        r'\bpl\b': 'place',
        r'\bsq\b': 'square',
        r'\bste\b': 'suite',
        r'\bapt\b': 'apartment'
    }
    for k, v in abbrevs.items():
        s = s.str.replace(k, v, regex=True)
    # Normalize whitespace
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s

def extract_features(df):
    df = df.copy()
    
    # Fill NA
    df['business_name'] = df['business_name'].fillna('')
    df['business_address'] = df['business_address'].fillna('')
    df['country'] = df['country'].fillna('')
    
    df['norm_name'] = normalize_text_series(df['business_name'])
    df['norm_address'] = normalize_text_series(df['business_address'])
    
    # Extract first alnum token of address
    df['address_first_token'] = df['norm_address'].str.split().str[0].fillna('')
    
    # Name length bucket
    name_len = df['norm_name'].str.len()
    df['name_len_bucket'] = pd.cut(name_len, bins=[-1, 5, 15, 30, 50, 1000], labels=[0, 1, 2, 3, 4], right=True).astype(int)
    
    # Combine normalized text for embeddings
    df['text_for_embedding'] = df['norm_name'] + " " + df['norm_address'] + " " + df['country'].str.lower()
    
    return df
