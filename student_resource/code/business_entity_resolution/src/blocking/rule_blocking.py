"""
Multi-Key Rule Blocking Module for Business Entity Resolution (Improved).
Generates candidate pairs per Source 1 entity using country, pincode, token, prefix, and bigram blocking keys.

IMPROVEMENTS over v1:
- Added bigram blocking key (first 2 tokens of core name joined)
- Added address token blocking key (first addr token for city-level grouping)
- Increased max_candidates_per_key and max_total_candidates for higher recall
- Randomize overflow selection to avoid systematic bias (shuffle before truncation)
- Added 6-char prefix key for longer names (French/English)
- Added pincode-prefix (first 3 digits) as a secondary geographic key
"""

import random
from collections import defaultdict
from typing import Dict, Set
import pandas as pd


def build_blocking_indices(df_s2: pd.DataFrame, df_s3: pd.DataFrame):
    """
    Build fast inverted indices for S2 and S3 records.
    Returns: pincode_idx, prefix4_idx, prefix6_idx, token_idx, bigram_idx, addr_token_idx, pin_prefix_idx
    """
    pincode_idx = defaultdict(list)
    prefix4_idx = defaultdict(list)
    prefix6_idx = defaultdict(list)
    token_idx = defaultdict(list)
    bigram_idx = defaultdict(list)
    addr_token_idx = defaultdict(list)
    pin_prefix_idx = defaultdict(list)

    df_combined = pd.concat([
        df_s2[["entity_id", "country_norm", "pincode", "name_core", "addr_token1"]],
        df_s3[["entity_id", "country_norm", "pincode", "name_core", "addr_token1"]],
    ], ignore_index=True)

    for row in df_combined.itertuples(index=False):
        eid = row.entity_id
        country = row.country_norm
        pin = row.pincode
        name = row.name_core
        addr_t1 = row.addr_token1

        # Key 1: Country + Pincode (exact)
        if pin:
            pincode_idx[(country, pin)].append(eid)
            # Key 7: Country + Pincode prefix (first 3 digits)
            if len(pin) >= 3:
                pin_prefix_idx[(country, pin[:3])].append(eid)

        # Key 2: Country + First 4 chars of core name
        if len(name) >= 3:
            prefix4_idx[(country, name[:4])].append(eid)

        # Key 3: Country + First 6 chars (for longer/French names)
        if len(name) >= 6:
            prefix6_idx[(country, name[:6])].append(eid)

        # Key 4: Country + Distinctive tokens (length >= 4, up to 4 tokens)
        tokens = [t for t in name.split() if len(t) >= 4][:4]
        for token in tokens:
            token_idx[(country, token)].append(eid)

        # Key 5: Country + First bigram (token[0]_token[1])
        name_tokens = name.split()
        if len(name_tokens) >= 2:
            bigram = name_tokens[0] + "_" + name_tokens[1]
            bigram_idx[(country, bigram)].append(eid)
        elif len(name_tokens) == 1:
            bigram_idx[(country, name_tokens[0])].append(eid)

        # Key 6: Country + First address token (city/area grouping)
        if addr_t1 and len(addr_t1) >= 3:
            addr_token_idx[(country, addr_t1)].append(eid)

    return pincode_idx, prefix4_idx, prefix6_idx, token_idx, bigram_idx, addr_token_idx, pin_prefix_idx


def _sample_candidates(matches, max_per_key: int) -> list:
    """Randomly sample if bucket is too large (avoids head bias)."""
    if len(matches) <= max_per_key:
        return matches
    return random.sample(matches, max_per_key)


def generate_rule_candidates(
    df_s1: pd.DataFrame,
    pincode_idx: dict,
    prefix4_idx: dict,
    prefix6_idx: dict,
    token_idx: dict,
    bigram_idx: dict,
    addr_token_idx: dict,
    pin_prefix_idx: dict,
    max_candidates_per_key: int = 75,
    max_total_candidates: int = 200,
) -> Dict[str, Set[str]]:
    """
    Fast query of inverted indices for each S1 entity.
    """
    candidates_by_s1 = {}
    rng = random.Random(42)

    for row in df_s1.itertuples(index=False):
        s1_id = row.entity_id
        country = row.country_norm
        pin = row.pincode
        name = row.name_core
        addr_t1 = row.addr_token1

        cands = set()

        # 1. Pincode exact
        if pin and (country, pin) in pincode_idx:
            cands.update(_sample_candidates(pincode_idx[(country, pin)], max_candidates_per_key))

        # 2. Prefix-4
        if len(name) >= 3:
            key = (country, name[:4])
            if key in prefix4_idx:
                cands.update(_sample_candidates(prefix4_idx[key], max_candidates_per_key))

        # 3. Prefix-6 (higher precision for longer names)
        if len(name) >= 6:
            key = (country, name[:6])
            if key in prefix6_idx:
                cands.update(_sample_candidates(prefix6_idx[key], max_candidates_per_key))

        # 4. Distinctive tokens
        tokens = [t for t in name.split() if len(t) >= 4][:4]
        for token in tokens:
            key = (country, token)
            if key in token_idx:
                cands.update(_sample_candidates(token_idx[key], max_candidates_per_key))

        # 5. Bigram
        name_tokens = name.split()
        if len(name_tokens) >= 2:
            bigram = name_tokens[0] + "_" + name_tokens[1]
            key = (country, bigram)
        elif len(name_tokens) == 1:
            key = (country, name_tokens[0])
        else:
            key = None
        if key and key in bigram_idx:
            cands.update(_sample_candidates(bigram_idx[key], max_candidates_per_key))

        # 6. Address token (city/area grouping)
        if addr_t1 and len(addr_t1) >= 3:
            key = (country, addr_t1)
            if key in addr_token_idx:
                cands.update(_sample_candidates(addr_token_idx[key], max_candidates_per_key))

        # 7. Pincode prefix (geographic fuzzy)
        if pin and len(pin) >= 3:
            key = (country, pin[:3])
            if key in pin_prefix_idx:
                cands.update(_sample_candidates(pin_prefix_idx[key], max_candidates_per_key))

        if len(cands) > max_total_candidates:
            candidates_by_s1[s1_id] = set(rng.sample(list(cands), max_total_candidates))
        else:
            candidates_by_s1[s1_id] = cands

    return candidates_by_s1
