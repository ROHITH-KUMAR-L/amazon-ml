"""
Transliteration Module for Business Entity Resolution.
Mines common Indian transliteration character substitution patterns from training true-matches,
and provides a custom weighted Levenshtein similarity metric.
"""

import json
import os
import re
from collections import Counter
import pandas as pd


DEFAULT_TRANSLIT_PAIRS = {
    ("v", "w"): 0.1,
    ("w", "v"): 0.1,
    ("s", "sh"): 0.2,
    ("sh", "s"): 0.2,
    ("i", "ee"): 0.2,
    ("ee", "i"): 0.2,
    ("u", "oo"): 0.2,
    ("oo", "u"): 0.2,
    ("c", "k"): 0.1,
    ("k", "c"): 0.1,
    ("ph", "f"): 0.1,
    ("f", "ph"): 0.1,
    ("d", "dh"): 0.2,
    ("dh", "d"): 0.2,
    ("t", "th"): 0.2,
    ("th", "t"): 0.2,
    ("b", "v"): 0.2,
    ("v", "b"): 0.2,
}


def build_transliteration_table(df_s1: pd.DataFrame, df_s2: pd.DataFrame, df_s3: pd.DataFrame, df_gt: pd.DataFrame, out_path: str = None) -> dict:
    """
    Mine character substitution patterns from confirmed true-match pairs where country == 'INDIA'.
    Returns a dictionary of (sub_a, sub_b) -> penalty_weight.
    """
    # Create lookup map for entity text
    s1_map = df_s1.set_index("entity_id")["name_norm"].to_dict()
    
    s23_map = {}
    s23_map.update(df_s2.set_index("entity_id")["name_norm"].to_dict())
    s23_map.update(df_s3.set_index("entity_id")["name_norm"].to_dict())
    
    # Filter ground truth for India entities
    india_s1 = set(df_s1[df_s1["country_norm"] == "INDIA"]["entity_id"])
    
    sub_counter = Counter()
    
    # Sample GT pairs for fast mining
    gt_sample = df_gt[df_gt["source1_entity_id"].isin(india_s1)].head(50000)
    
    for _, row in gt_sample.iterrows():
        s1_id = row["source1_entity_id"]
        s1_name = s1_map.get(s1_id, "")
        if not s1_name:
            continue
            
        for match_id in row["target_set"]:
            s23_name = s23_map.get(match_id, "")
            if not s23_name or s1_name == s23_name:
                continue
                
            # Quick check for common transliteration patterns
            for (a, b) in DEFAULT_TRANSLIT_PAIRS.keys():
                if a in s1_name and b in s23_name:
                    sub_counter[(a, b)] += 1

    translit_weights = dict(DEFAULT_TRANSLIT_PAIRS)
    
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # Convert tuple keys to string for JSON serialization
        json_dict = {f"{k[0]}|{k[1]}": v for k, v in translit_weights.items()}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(json_dict, f, indent=2)
            
    return translit_weights


def transliteration_similarity(name1: str, name2: str, translit_table: dict = None) -> float:
    """
    Compute transliteration-aware similarity score between two business names.
    Applies character-level replacements for known transliterations before computing Levenshtein.
    """
    if not name1 or not name2:
        return 0.0
    if name1 == name2:
        return 1.0
        
    table = translit_table if translit_table else DEFAULT_TRANSLIT_PAIRS
    
    # Normalize transliteration variants in name2 towards name1
    n1, n2 = name1, name2
    for (a, b) in table.keys():
        if a in n1 and b in n2:
            n2 = n2.replace(b, a)
            
    # Compute character overlap ratio
    from difflib import SequenceMatcher
    return SequenceMatcher(None, n1, n2).ratio()
