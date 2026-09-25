"""
TF-IDF / Character N-Gram Candidate Retrieval Module for Business Entity Resolution.
Uses country-partitioned TF-IDF vectorization over character 3-grams to retrieve top-K candidates.
"""

from typing import Dict, Set
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def retrieve_tfidf_candidates(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    top_k: int = 15,
    max_features: int = 100000
) -> Dict[str, Set[str]]:
    """
    Retrieve top-K candidate S2/S3 entities for each S1 entity using character 3-gram TF-IDF cosine similarity,
    partitioned by country.
    """
    candidates_by_s1 = {}
    
    # Combine S2 and S3 for indexing
    df_s23 = pd.concat([
        df_s2[["entity_id", "country_norm", "name_norm", "address_norm"]],
        df_s3[["entity_id", "country_norm", "name_norm", "address_norm"]]
    ], ignore_index=True)
    
    # Combine name and address into a single text representation
    df_s1["text_rep"] = (df_s1["name_norm"] + " " + df_s1["address_norm"]).str.strip()
    df_s23["text_rep"] = (df_s23["name_norm"] + " " + df_s23["address_norm"]).str.strip()
    
    countries = df_s1["country_norm"].unique()
    
    for country in countries:
        s1_country = df_s1[df_s1["country_norm"] == country]
        s23_country = df_s23[df_s23["country_norm"] == country]
        
        if len(s1_country) == 0 or len(s23_country) == 0:
            continue
            
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 3),
            max_features=max_features,
            sublinear_tf=True
        )
        
        # Fit vectorizer on combined corpus
        corpus = pd.concat([s1_country["text_rep"], s23_country["text_rep"]])
        vectorizer.fit(corpus)
        
        X_s1 = vectorizer.transform(s1_country["text_rep"])
        X_s23 = vectorizer.transform(s23_country["text_rep"])
        
        s1_ids = s1_country["entity_id"].values
        s23_ids = s23_country["entity_id"].values
        
        # Batch processing matrix multiplication to prevent memory overflow
        batch_size = 50000
        n_s1 = X_s1.shape[0]
        
        for start_idx in range(0, n_s1, batch_size):
            end_idx = min(start_idx + batch_size, n_s1)
            batch_s1 = X_s1[start_idx:end_idx]
            
            # Compute cosine similarity matrix (sparse product)
            sim_matrix = batch_s1.dot(X_s23.T)
            
            for i in range(batch_s1.shape[0]):
                s1_id = s1_ids[start_idx + i]
                row_data = sim_matrix[i]
                
                if row_data.nnz == 0:
                    candidates_by_s1[s1_id] = set()
                    continue
                    
                # Extract top_k highest similarity indices
                data = row_data.data
                indices = row_data.indices
                
                if len(data) > top_k:
                    top_indices = indices[np.argpartition(data, -top_k)[-top_k:]]
                else:
                    top_indices = indices
                    
                candidates_by_s1[s1_id] = set(s23_ids[top_indices])
                
    return candidates_by_s1
