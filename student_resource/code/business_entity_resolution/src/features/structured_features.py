"""
Structured & Categorical Feature Extraction Module for Business Entity Resolution.
Computes PIN code, street number, legal suffix, landmark, and missingness match features.
"""

import pandas as pd


def compute_structured_pair_features(row_s1, row_s23) -> dict:
    """
    Compute structured comparison features for a pair of entities (S1, S2/S3).
    """
    country_match = int(row_s1["country_norm"] == row_s23["country_norm"])
    
    # Pincode features
    pin1 = row_s1["pincode"]
    pin2 = row_s23["pincode"]
    
    has_pin1 = row_s1["has_pincode"]
    has_pin2 = row_s23["has_pincode"]
    both_pin = int(has_pin1 and has_pin2)
    one_pin = int(has_pin1 or has_pin2) - both_pin
    
    if both_pin:
        pin_exact = int(pin1 == pin2)
        pin_prefix = int(pin1[:3] == pin2[:3]) if len(pin1) >= 3 and len(pin2) >= 3 else 0
    else:
        pin_exact = 0
        pin_prefix = 0
        
    # Street number features
    st1 = row_s1["street_num"]
    st2 = row_s23["street_num"]
    st_match = int(bool(st1 and st2 and st1 == st2))
    
    # Legal suffix features
    suf1 = row_s1["legal_suffix"]
    suf2 = row_s23["legal_suffix"]
    suffix_match = int(bool(suf1 and suf2 and suf1 == suf2))
    suffix_mismatch = int(bool(suf1 and suf2 and suf1 != suf2))
    
    # Missingness features
    has_addr1 = row_s1["has_address"]
    has_addr2 = row_s23["has_address"]
    both_has_addr = int(has_addr1 and has_addr2)
    
    return {
        "country_match": country_match,
        "pincode_exact_match": pin_exact,
        "pincode_prefix_match": pin_prefix,
        "both_has_pincode": both_pin,
        "one_missing_pincode": one_pin,
        "street_num_match": st_match,
        "suffix_match": suffix_match,
        "suffix_mismatch": suffix_mismatch,
        "both_has_address": both_has_addr,
        "has_landmark_s1": row_s1["has_landmark"],
        "has_landmark_s23": row_s23["has_landmark"],
    }
