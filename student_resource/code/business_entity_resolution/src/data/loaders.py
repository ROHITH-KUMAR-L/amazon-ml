"""
Data Loaders Module for Business Entity Resolution.
Handles robust loading of tab-separated datasets and ground truth mappings.
"""

import os
import pandas as pd

def load_source_tsv(file_path: str) -> pd.DataFrame:
    """
    Load a source TSV file (source1, source2, or source3) with explicit tab separator,
    imputing missing values safely.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found: {file_path}")
    
    df = pd.read_csv(
        file_path,
        sep="\t",
        dtype={"entity_id": str, "business_name": str, "business_address": str, "country": str},
        keep_default_na=False
    )
    
    # Ensure strings and strip whitespace
    df["entity_id"] = df["entity_id"].astype(str).str.strip()
    df["business_name"] = df["business_name"].fillna("").astype(str).str.strip()
    df["business_address"] = df["business_address"].fillna("").astype(str).str.strip()
    df["country"] = df["country"].fillna("").astype(str).str.strip()
    
    return df


def load_ground_truth(file_path: str) -> pd.DataFrame:
    """
    Load ground truth TSV mapping source1_entity_id -> matched_entity_ids (comma-separated).
    Returns a DataFrame with parsed sets of matched IDs.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ground truth file not found: {file_path}")
    
    df = pd.read_csv(
        file_path,
        sep="\t",
        dtype={"source1_entity_id": str, "matched_entity_ids": str},
        keep_default_na=False
    )
    
    df["source1_entity_id"] = df["source1_entity_id"].astype(str).str.strip()
    df["matched_entity_ids"] = df["matched_entity_ids"].fillna("").astype(str).str.strip()
    
    # Parse comma-separated IDs into sets
    def parse_ids(val: str):
        if not val:
            return set()
        return set(x.strip() for x in val.split(",") if x.strip())
    
    df["target_set"] = df["matched_entity_ids"].apply(parse_ids)
    return df


def load_all_sources(data_dir: str, prefix: str = "train"):
    """
    Load source1, source2, source3 for a given prefix ('train' or 'test').
    Returns (df_s1, df_s2, df_s3).
    """
    s1_path = os.path.join(data_dir, f"{prefix}_source1.tsv")
    s2_path = os.path.join(data_dir, f"{prefix}_source2.tsv")
    s3_path = os.path.join(data_dir, f"{prefix}_source3.tsv")
    
    df_s1 = load_source_tsv(s1_path)
    df_s2 = load_source_tsv(s2_path)
    df_s3 = load_source_tsv(s3_path)
    
    return df_s1, df_s2, df_s3
