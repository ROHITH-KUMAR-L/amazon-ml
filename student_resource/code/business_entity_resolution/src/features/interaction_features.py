"""
Feature Extraction Orchestrator & Cross-Field Interaction Module (Improved).

IMPROVEMENTS over v1:
- Added token_sort_ratio for name & core (word-order invariant)
- Added char 3-gram Jaccard for name and address
- Added name-length ratio feature (detects abbreviations like "ABC" vs "ABC Corp Ltd")
- Added name_sorted similarity (sorted tokens)
- Added addr_idf_jaccard for address IDF overlap
- Added ratio features: name_len_ratio, name_ntoken_ratio
- Added composite features with more weight on IDF-Jaccard
- Separated name_sort (token_sort_ratio) from name_set (token_set_ratio)
- Added addr_token_match (first address token exact match — city level)
- Fixed: name_sort was duplicate of name_set — now properly uses token_sort_ratio
- Added has_country_mismatch flag
- Better rank features: percentile rank, normalized gap
"""

from typing import Dict, List, Tuple
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features.idf_jaccard import IDFEstimator
from src.data.transliteration import transliteration_similarity


def _seq_ratio(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    return SequenceMatcher(None, s1, s2).ratio()


def _token_sort_ratio(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    t1 = " ".join(sorted(s1.split()))
    t2 = " ".join(sorted(s2.split()))
    return SequenceMatcher(None, t1, t2).ratio()


def _token_set_jaccard(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    t1 = set(s1.split())
    t2 = set(s2.split())
    inter = t1 & t2
    union = t1 | t2
    return len(inter) / len(union) if union else 0.0


def _char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    if len(s1) < n or len(s2) < n:
        return 0.0
    g1 = set(s1[i:i+n] for i in range(len(s1) - n + 1))
    g2 = set(s2[i:i+n] for i in range(len(s2) - n + 1))
    union = g1 | g2
    return len(g1 & g2) / len(union) if union else 0.0


def _len_ratio(s1: str, s2: str) -> float:
    l1, l2 = len(s1), len(s2)
    if l1 == 0 and l2 == 0:
        return 1.0
    if l1 == 0 or l2 == 0:
        return 0.0
    return min(l1, l2) / max(l1, l2)


def build_candidate_feature_matrix(
    candidate_pairs: List[Tuple[str, str]],
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    idf_estimator: IDFEstimator,
    translit_table: dict = None,
) -> pd.DataFrame:
    """
    Build complete feature matrix for candidate pairs.
    """
    if not candidate_pairs:
        return pd.DataFrame()

    # Pre-index entity attributes
    COLS = [
        "name_norm", "name_core", "name_sorted", "core_sorted",
        "address_norm", "addr_token1",
        "pincode", "street_num", "legal_suffix",
        "country_norm",
        "has_pincode", "has_address", "has_landmark",
        "name_len", "name_ntoken",
    ]

    s1_dict = df_s1.set_index("entity_id")[COLS].to_dict("index")

    s23_dict = {}
    s23_dict.update(df_s2.set_index("entity_id")[COLS].to_dict("index"))
    s23_dict.update(df_s3.set_index("entity_id")[COLS].to_dict("index"))

    records = []

    for s1_id, cand_id in candidate_pairs:
        r1 = s1_dict.get(s1_id)
        r2 = s23_dict.get(cand_id)

        if not r1 or not r2:
            continue

        n1, n2 = r1["name_norm"], r2["name_norm"]
        c1, c2 = r1["name_core"], r2["name_core"]
        ns1, ns2 = r1["name_sorted"], r2["name_sorted"]
        cs1, cs2 = r1["core_sorted"], r2["core_sorted"]
        a1, a2 = r1["address_norm"], r2["address_norm"]
        at1, at2 = r1["addr_token1"], r2["addr_token1"]

        # ── String similarities ──────────────────────────────────────────────
        name_lev = _seq_ratio(n1, n2)
        name_sort = _token_sort_ratio(n1, n2)
        name_set = _token_set_jaccard(n1, n2)
        name_char3 = _char_ngram_jaccard(n1, n2, 3)

        core_lev = _seq_ratio(c1, c2)
        core_sort = _token_sort_ratio(c1, c2)
        core_set = _token_set_jaccard(c1, c2)
        core_sorted_sim = _seq_ratio(cs1, cs2)

        addr_lev = _seq_ratio(a1, a2)
        addr_set = _token_set_jaccard(a1, a2)
        addr_char3 = _char_ngram_jaccard(a1, a2, 3)

        # Addr first token exact match (city-level)
        addr_token_match = int(bool(at1 and at2 and at1 == at2))

        # ── IDF-weighted features ────────────────────────────────────────────
        name_idf = idf_estimator.weighted_jaccard(n1, n2)
        core_idf = idf_estimator.weighted_jaccard(c1, c2)
        addr_idf = idf_estimator.weighted_jaccard(a1, a2)

        # ── Transliteration (India) ──────────────────────────────────────────
        is_india = (r1["country_norm"] == "INDIA" or r2["country_norm"] == "INDIA")
        translit_sim = transliteration_similarity(c1, c2, translit_table) if is_india else core_lev

        # ── Length / token count ratio ───────────────────────────────────────
        name_len_ratio = _len_ratio(n1, n2)
        core_len_ratio = _len_ratio(c1, c2)
        name_ntoken_ratio = _len_ratio(
            " ".join(["x"] * max(r1["name_ntoken"], 1)),
            " ".join(["x"] * max(r2["name_ntoken"], 1)),
        )

        # ── Structured features ──────────────────────────────────────────────
        country_match = int(r1["country_norm"] == r2["country_norm"])
        country_mismatch = int(not country_match)

        pin1, pin2 = r1["pincode"], r2["pincode"]
        both_pin = int(bool(r1["has_pincode"]) and bool(r2["has_pincode"]))
        pin_exact = int(both_pin and pin1 == pin2)
        pin_prefix = int(both_pin and len(pin1) >= 3 and len(pin2) >= 3 and pin1[:3] == pin2[:3])

        st1, st2 = r1["street_num"], r2["street_num"]
        st_match = int(bool(st1 and st2 and st1 == st2))

        suf1, suf2 = r1["legal_suffix"], r2["legal_suffix"]
        suf_match = int(bool(suf1 and suf2 and suf1 == suf2))
        suf_mismatch = int(bool(suf1 and suf2 and suf1 != suf2))

        # ── Interaction features ─────────────────────────────────────────────
        name_addr_prod = name_set * addr_set
        name_idf_addr_prod = name_idf * addr_set
        high_name_pin_mismatch = int(name_set > 0.8 and both_pin and not pin_exact)
        # "name match but address mismatch" signal
        name_match_addr_mismatch = int(name_set > 0.7 and addr_set < 0.2)

        rec = {
            "s1_id": s1_id,
            "cand_id": cand_id,
            # Name features
            "name_lev": name_lev,
            "name_sort": name_sort,
            "name_set": name_set,
            "name_char3": name_char3,
            "name_idf_jaccard": name_idf,
            "name_len_ratio": name_len_ratio,
            "name_ntoken_ratio": name_ntoken_ratio,
            # Core name features
            "core_lev": core_lev,
            "core_sort": core_sort,
            "core_set": core_set,
            "core_sorted_sim": core_sorted_sim,
            "core_idf": core_idf,
            "core_len_ratio": core_len_ratio,
            # Address features
            "addr_lev": addr_lev,
            "addr_set": addr_set,
            "addr_char3": addr_char3,
            "addr_idf_jaccard": addr_idf,
            "addr_token_match": addr_token_match,
            # Transliteration
            "translit_sim": translit_sim,
            # Structured
            "country_match": country_match,
            "country_mismatch": country_mismatch,
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
            # Interactions
            "name_addr_prod": name_addr_prod,
            "name_idf_addr_prod": name_idf_addr_prod,
            "high_name_pin_mismatch": high_name_pin_mismatch,
            "name_match_addr_mismatch": name_match_addr_mismatch,
        }
        records.append(rec)

    df_feats = pd.DataFrame(records)
    if df_feats.empty:
        return df_feats

    # ── Composite similarity ─────────────────────────────────────────────────
    df_feats["composite_sim"] = (
        0.35 * df_feats["name_idf_jaccard"]
        + 0.20 * df_feats["core_lev"]
        + 0.15 * df_feats["name_sort"]
        + 0.15 * df_feats["addr_set"]
        + 0.10 * df_feats["pincode_exact_match"]
        + 0.05 * df_feats["translit_sim"]
    )

    # ── Neighbourhood / rank features per S1 entity ──────────────────────────
    grp = df_feats.groupby("s1_id")["composite_sim"]
    df_feats["cand_rank"] = grp.rank(ascending=False, method="min")
    df_feats["max_sim_for_s1"] = grp.transform("max")
    df_feats["min_sim_for_s1"] = grp.transform("min")
    df_feats["mean_sim_for_s1"] = grp.transform("mean")
    df_feats["cand_count_for_s1"] = grp.transform("count")
    df_feats["sim_gap_to_best"] = df_feats["max_sim_for_s1"] - df_feats["composite_sim"]
    df_feats["sim_percentile"] = df_feats["cand_rank"] / df_feats["cand_count_for_s1"]
    df_feats["sim_gap_norm"] = df_feats["sim_gap_to_best"] / (
        df_feats["max_sim_for_s1"] - df_feats["min_sim_for_s1"] + 1e-8
    )

    return df_feats
