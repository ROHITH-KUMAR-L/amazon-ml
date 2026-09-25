"""
Normalization Module for Business Entity Resolution (Optimized Vectorized Version).
Applies NFKC normalization, legal suffix mapping, address parsing, and missingness flags.
"""

import re
import unicodedata
import pandas as pd

LEGAL_SUFFIX_RE = r"\b(corporation|corp|incorporated|inc|private|pvt|limited|ltd|llc|plc|company|co|enterprises?|services?|solutions?|traders?|trading|stores?)\b"

SUFFIX_MAP = {
    "corporation": "corp", "corp": "corp",
    "incorporated": "inc", "inc": "inc",
    "private": "pvt", "pvt": "pvt",
    "limited": "ltd", "ltd": "ltd",
    "llc": "llc", "plc": "plc",
    "company": "co", "co": "co",
    "enterprise": "ent", "enterprises": "ent",
    "service": "svc", "services": "svc",
    "solution": "sol", "solutions": "sol",
    "trader": "trader", "traders": "trader", "trading": "trader",
    "store": "store", "stores": "store"
}


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized fast normalization & parsing.
    """
    df = df.copy()

    # 1. Missingness flags
    df["has_name"] = (df["business_name"].str.len() > 0).astype(int)
    df["has_address"] = (df["business_address"].str.len() > 0).astype(int)

    # 2. Text normalization
    name_str = df["business_name"].str.lower()
    name_str = name_str.str.replace("&", " and ", regex=False)
    name_str = name_str.str.replace(r"[^\w\s]", " ", regex=True)
    name_str = name_str.str.replace(r"\s+", " ", regex=True).str.strip()
    df["name_norm"] = name_str

    # Extract legal suffix
    extracted_suffix = name_str.str.extract(LEGAL_SUFFIX_RE, expand=False).fillna("")
    df["legal_suffix"] = extracted_suffix.map(SUFFIX_MAP).fillna("")
    df["name_core"] = name_str.str.replace(LEGAL_SUFFIX_RE, "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()

    # 3. Address normalization & parsing
    addr_str = df["business_address"].str.lower()
    addr_str = addr_str.str.replace("&", " and ", regex=False)
    addr_str = addr_str.str.replace(r"[^\w\s]", " ", regex=True)
    addr_str = addr_str.str.replace(r"\s+", " ", regex=True).str.strip()
    df["address_norm"] = addr_str

    # Vectorized regex extractions for address components
    df["pincode"] = addr_str.str.extract(r"\b(\d{5,6})\b", expand=False).fillna("")
    df["has_pincode"] = (df["pincode"].str.len() > 0).astype(int)
    df["street_num"] = addr_str.str.extract(r"\b(\d+[a-z]?)\b", expand=False).fillna("")
    df["has_landmark"] = addr_str.str.contains(r"\b(?:near|opposite|opp|behind|above|next to|beside|adj)\b", regex=True).astype(int)

    # 4. Open-set country normalization
    df["country_norm"] = df["country"].str.strip().str.upper()

    return df
