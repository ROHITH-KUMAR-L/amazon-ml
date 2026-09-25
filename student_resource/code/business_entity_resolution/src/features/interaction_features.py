"""
Feature Extraction Orchestrator & Cross-Field Interaction Module (Optimized Fast Version).
Builds feature matrix for candidate pairs using batch vectorized string operations.
"""

from typing import Dict, List, Tuple
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features.idf_jaccard import IDFEstimator
from src.data.transliteration import transliteration_similarity


def fast_levenshtein(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    return SequenceMatcher(None, s1, s2).ratio()


def fast_token_set(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    t1 = set(s1.split())
    t2 = set(s2.split())
    inter = t1.intersection(t2)
    if not inter:
        return 0.0
    union = t1.union(t2)
    return len(inter) / len(union)


def build_candidate_feature_matrix(
    candidate_pairs: List[Tuple[str, str]],
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    idf_estimator: IDFEstimator,
    translit_table: dict = None
) -> pd.DataFrame:
    """
    Build complete feature matrix for candidate pairs.
    """
    if not candidate_pairs:
        return pd.DataFrame()

    # Pre-index entity attributes
    s1_dict = df_s1.set_index("entity_id")[
        ["name_norm", "name_core", "address_norm", "pincode", "street_num", "legal_suffix", "country_norm", "has_pincode", "has_address", "has_landmark"]
    ].to_dict("index")

    s23_dict = {}
    s23_dict.update(df_s2.set_index("entity_id")[
        ["name_norm", "name_core", "address_norm", "pincode", "street_num", "legal_suffix", "country_norm", "has_pincode", "has_address", "has_landmark"]
    ].to_dict("index"))
    s23_dict.update(df_s3.set_index("entity_id")[
        ["name_norm", "name_core", "address_norm", "pincode", "street_num", "legal_suffix", "country_norm", "has_pincode", "has_address", "has_landmark"]
    ].to_dict("index"))

    records = []

    for s1_id, cand_id in candidate_pairs:
        r1 = s1_dict.get(s1_id)
        r2 = s23_dict.get(cand_id)

        if not r1 or not r2:
            continue

        n1, n2 = r1["name_norm"], r2["name_norm"]
        c1, c2 = r1["name_core"], r2["name_core"]
        a1, a2 = r1["address_norm"], r2["address_norm"]

        # String similarities
        name_lev = fast_levenshtein(n1, n2)
        name_set = fast_token_set(n1, n2)
        core_lev = fast_levenshtein(c1, c2)
        addr_lev = fast_levenshtein(a1, a2)
        addr_set = fast_token_set(a1, a2)

        # IDF weighted features
        name_idf = idf_estimator.weighted_jaccard(n1, n2)
        addr_idf = idf_estimator.weighted_jaccard(a1, a2)

        # Transliteration similarity
        if r1["country_norm"] == "INDIA" or r2["country_norm"] == "INDIA":
            translit_sim = transliteration_similarity(c1, c2, translit_table)
        else:
            translit_sim = name_lev

        # Structured features
        country_match = int(r1["country_norm"] == r2["country_norm"])
        pin1, pin2 = r1["pincode"], r2["pincode"]
        both_pin = int(r1["has_pincode"] and r2["has_pincode"])
        pin_exact = int(both_pin and pin1 == pin2)
        pin_prefix = int(both_pin and pin1[:3] == pin2[:3] and len(pin1) >= 3)

        st1, st2 = r1["street_num"], r2["street_num"]
        st_match = int(bool(st1 and st2 and st1 == st2))

        suf1, suf2 = r1["legal_suffix"], r2["legal_suffix"]
        suf_match = int(bool(suf1 and suf2 and suf1 == suf2))
        suf_mismatch = int(bool(suf1 and suf2 and suf1 != suf2))

        # Interactions
        name_addr_prod = name_set * addr_set
        name_country_prod = name_set * country_match
        high_name_pin_mismatch = int(name_set > 0.8 and both_pin and not pin_exact)

        rec = {
            "s1_id": s1_id,
            "cand_id": cand_id,
            "name_lev": name_lev,
            "name_sort": name_set,
            "name_set": name_set,
            "core_lev": core_lev,
            "addr_lev": addr_lev,
            "addr_sort": addr_set,
            "addr_set": addr_set,
            "char3_sim": 0.5 * (name_set + addr_set),
            "name_idf_jaccard": name_idf,
            "addr_idf_jaccard": addr_idf,
            "translit_sim": translit_sim,
            "name_addr_prod": name_addr_prod,
            "name_country_prod": name_country_prod,
            "high_name_pin_mismatch": high_name_pin_mismatch,
            "country_match": country_match,
            "pincode_exact_match": pin_exact,
            "pincode_prefix_match": pin_prefix,
            "both_has_pincode": both_pin,
            "one_missing_pincode": int(r1["has_pincode"] != r2["has_pincode"]),
            "street_num_match": st_match,
            "suffix_match": suf_match,
            "suffix_mismatch": suf_mismatch,
            "both_has_address": int(r1["has_address"] and r2["has_address"]),
            "has_landmark_s1": r1["has_landmark"],
            "has_landmark_s23": r2["has_landmark"],
        }
        records.append(rec)

    df_feats = pd.DataFrame(records)
    if df_feats.empty:
        return df_feats

    # Neighborhood / Rank features per S1 entity
    df_feats["composite_sim"] = 0.5 * df_feats["name_set"] + 0.3 * df_feats["addr_set"] + 0.2 * df_feats["name_idf_jaccard"]
    df_feats["cand_rank"] = df_feats.groupby("s1_id")["composite_sim"].rank(ascending=False, method="min")
    df_feats["max_sim_for_s1"] = df_feats.groupby("s1_id")["composite_sim"].transform("max")
    df_feats["cand_count_for_s1"] = df_feats.groupby("s1_id")["composite_sim"].transform("count")
    df_feats["sim_gap_to_best"] = df_feats["max_sim_for_s1"] - df_feats["composite_sim"]

    return df_feats
