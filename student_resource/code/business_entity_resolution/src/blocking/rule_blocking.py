"""
Multi-Key Rule Blocking Module for Business Entity Resolution (Optimized Fast Version).
Generates candidate pairs per Source 1 entity using country, pincode, token, and prefix blocking keys.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd


def build_blocking_indices(df_s2: pd.DataFrame, df_s3: pd.DataFrame):
    """
    Build fast inverted indices for S2 and S3 records using namedtuples iteration.
    """
    pincode_idx = defaultdict(list)
    prefix_idx = defaultdict(list)
    token_idx = defaultdict(list)

    # Combined S2 and S3 records
    df_combined = pd.concat([
        df_s2[["entity_id", "country_norm", "pincode", "name_core"]],
        df_s3[["entity_id", "country_norm", "pincode", "name_core"]]
    ], ignore_index=True)

    for row in df_combined.itertuples(index=False):
        eid = row.entity_id
        country = row.country_norm
        pin = row.pincode
        name = row.name_core

        # Key 1: Country + Pincode
        if pin:
            pincode_idx[(country, pin)].append(eid)

        # Key 2: Country + First 4 chars of core name
        if len(name) >= 3:
            prefix = name[:4]
            prefix_idx[(country, prefix)].append(eid)

        # Key 3: Country + Distinctive tokens (length >= 4)
        tokens = [t for t in name.split() if len(t) >= 4][:3]
        for token in tokens:
            token_idx[(country, token)].append(eid)

    return pincode_idx, prefix_idx, token_idx


def generate_rule_candidates(
    df_s1: pd.DataFrame,
    pincode_idx: dict,
    prefix_idx: dict,
    token_idx: dict,
    max_candidates_per_key: int = 50,
    max_total_candidates: int = 100
) -> Dict[str, Set[str]]:
    """
    Fast query of inverted indices for each S1 entity using itertuples.
    """
    candidates_by_s1 = {}

    for row in df_s1.itertuples(index=False):
        s1_id = row.entity_id
        country = row.country_norm
        pin = row.pincode
        name = row.name_core

        cands = set()

        # 1. Query Pincode Index
        if pin and (country, pin) in pincode_idx:
            matches = pincode_idx[(country, pin)]
            if len(matches) <= max_candidates_per_key:
                cands.update(matches)
            else:
                cands.update(matches[:max_candidates_per_key])

        # 2. Query Prefix Index
        if len(name) >= 3:
            prefix = name[:4]
            if (country, prefix) in prefix_idx:
                matches = prefix_idx[(country, prefix)]
                if len(matches) <= max_candidates_per_key:
                    cands.update(matches)
                else:
                    cands.update(matches[:max_candidates_per_key])

        # 3. Query Token Index
        tokens = [t for t in name.split() if len(t) >= 4][:3]
        for token in tokens:
            if (country, token) in token_idx:
                matches = token_idx[(country, token)]
                if len(matches) <= max_candidates_per_key:
                    cands.update(matches)
                else:
                    cands.update(matches[:max_candidates_per_key])

        if len(cands) > max_total_candidates:
            candidates_by_s1[s1_id] = set(list(cands)[:max_total_candidates])
        else:
            candidates_by_s1[s1_id] = cands

    return candidates_by_s1
