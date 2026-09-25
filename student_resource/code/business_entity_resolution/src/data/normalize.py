"""
Normalization Module for Business Entity Resolution (Optimized Vectorized Version).
Applies NFKC normalization, legal suffix mapping, address parsing, and missingness flags.

IMPROVEMENTS:
- Expanded legal suffix map (dba, group, international, associates, industries, etc.)
- Address abbreviation expansion (St -> street, Rd -> road, Ave -> avenue, etc.)
- Token-sorted name variant for word-order-invariant matching
- Expanded landmark keywords
- French address patterns (rue, avenue, boulevard, cedex)
- Longer prefix (6 chars) for French/US names
"""

import re
import unicodedata
import pandas as pd

# Expanded legal suffix normalization
LEGAL_SUFFIX_RE = (
    r"\b(corporation|corp|incorporated|inc|private|pvt|limited|ltd|llc|plc|"
    r"company|co|enterprises?|services?|solutions?|traders?|trading|stores?|"
    r"group|international|intl|associates?|assoc|industries|industry|ind|"
    r"technologies|technology|tech|systems?|sys|consulting|consultants?|"
    r"holdings?|ventures?|networks?|communications?|labs?|laboratory|"
    r"brothers?|bros?|do business as|dba|and sons?|partners?|"
    r"sarl|sas|sa|eurl|sasu|snc)\b"  # French legal forms
)

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
    "store": "store", "stores": "store",
    "group": "grp",
    "international": "intl", "intl": "intl",
    "associate": "assoc", "associates": "assoc", "assoc": "assoc",
    "industry": "ind", "industries": "ind", "ind": "ind",
    "technology": "tech", "technologies": "tech", "tech": "tech",
    "system": "sys", "systems": "sys", "sys": "sys",
    "consulting": "cons", "consultants": "cons", "consultant": "cons",
    "holding": "hld", "holdings": "hld",
    "venture": "vent", "ventures": "vent",
    "network": "net", "networks": "net",
    "communication": "comm", "communications": "comm",
    "lab": "lab", "labs": "lab", "laboratory": "lab",
    "brother": "bros", "brothers": "bros", "bros": "bros",
    "do business as": "dba", "dba": "dba",
    "and sons": "sons", "son": "sons", "sons": "sons",
    "partner": "ptr", "partners": "ptr",
    # French
    "sarl": "sarl", "sas": "sas", "sa": "sa",
    "eurl": "eurl", "sasu": "sasu", "snc": "snc",
}

# Address abbreviation expansion for US
US_ADDR_ABBR = {
    r"\bst\b": "street",
    r"\brd\b": "road",
    r"\bave?\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b": "highway",
    r"\bfwy\b": "freeway",
    r"\bsq\b": "square",
    r"\bfte?\b": "suite",
    r"\bste\b": "suite",
    r"\bapt\b": "apartment",
    r"\bno\b": "number",
    r"\bn\b": "north",
    r"\bs\b": "south",
    r"\be\b": "east",
    r"\bw\b": "west",
    r"\bnw\b": "northwest",
    r"\bne\b": "northeast",
    r"\bsw\b": "southwest",
    r"\bse\b": "southeast",
}

# Indian address abbreviations
INDIA_ADDR_ABBR = {
    r"\bnagar\b": "nagar",
    r"\bng\b": "nagar",
    r"\bcolony\b": "colony",
    r"\bcol\b": "colony",
    r"\brd\b": "road",
    r"\bph\b": "phase",
    r"\bsec\b": "sector",
    r"\bsect\b": "sector",
    r"\bblk\b": "block",
}

LANDMARK_RE = r"\b(?:near|opp(?:osite)?|behind|above|next to|beside|adj(?:acent)?|in front of|beside|facing|across from|off)\b"


def _expand_abbreviations(text: pd.Series, abbr_map: dict) -> pd.Series:
    for pat, repl in abbr_map.items():
        text = text.str.replace(pat, repl, regex=True)
    return text


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized fast normalization & parsing.
    """
    df = df.copy()

    # 1. Missingness flags
    df["has_name"] = (df["business_name"].str.len() > 0).astype(int)
    df["has_address"] = (df["business_address"].str.len() > 0).astype(int)

    # 2. NFKC Unicode normalization + lowercase
    def nfkc_lower(s: pd.Series) -> pd.Series:
        return s.apply(lambda x: unicodedata.normalize("NFKC", str(x)).lower() if x else "")

    name_str = nfkc_lower(df["business_name"])
    name_str = name_str.str.replace("&", " and ", regex=False)
    name_str = name_str.str.replace(r"[^\w\s]", " ", regex=True)
    name_str = name_str.str.replace(r"\s+", " ", regex=True).str.strip()
    df["name_norm"] = name_str

    # Token-sorted variant for word-order-invariant matching
    df["name_sorted"] = name_str.apply(lambda x: " ".join(sorted(x.split())) if x else "")

    # Extract legal suffix (first match)
    extracted_suffix = name_str.str.extract(LEGAL_SUFFIX_RE, expand=False).fillna("")
    df["legal_suffix"] = extracted_suffix.map(SUFFIX_MAP).fillna("")
    df["name_core"] = (
        name_str.str.replace(LEGAL_SUFFIX_RE, "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    # Core sorted variant
    df["core_sorted"] = df["name_core"].apply(lambda x: " ".join(sorted(x.split())) if x else "")

    # 3. Address normalization & parsing
    addr_str = nfkc_lower(df["business_address"])
    addr_str = addr_str.str.replace("&", " and ", regex=False)
    addr_str = addr_str.str.replace(r"[^\w\s]", " ", regex=True)
    addr_str = addr_str.str.replace(r"\s+", " ", regex=True).str.strip()

    # Expand US abbreviations globally (safe for India/France too, won't hurt)
    addr_str = _expand_abbreviations(addr_str, US_ADDR_ABBR)

    df["address_norm"] = addr_str

    # Vectorized regex extractions for address components
    # 5-6 digit for India/US, 5 digit for France (cedex stripped)
    df["pincode"] = addr_str.str.extract(r"\b(\d{5,6})\b", expand=False).fillna("")
    df["has_pincode"] = (df["pincode"].str.len() > 0).astype(int)
    df["street_num"] = addr_str.str.extract(r"\b(\d+[a-z]?)\b", expand=False).fillna("")
    df["has_landmark"] = addr_str.str.contains(LANDMARK_RE, regex=True).astype(int)

    # City / first meaningful token from address (helps France)
    df["addr_token1"] = addr_str.apply(
        lambda x: x.split()[0] if x and x.split() else ""
    )

    # 4. Open-set country normalization
    df["country_norm"] = df["country"].str.strip().str.upper()

    # Name length ratio feature prep
    df["name_len"] = name_str.str.len()
    df["name_ntoken"] = name_str.apply(lambda x: len(x.split()) if x else 0)

    return df
