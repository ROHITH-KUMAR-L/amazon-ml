"""
Global Consistency & Submission Export Module for Business Entity Resolution.
Enforces 1-to-many bipartite matching constraints and outputs matching_results.tsv.
"""

import os
from typing import Dict, Set
import numpy as np
import pandas as pd


def resolve_global_assignments(
    df_feats: pd.DataFrame,
    pred_probas: np.ndarray,
    threshold: float
) -> Dict[str, Set[str]]:
    """
    Apply greedy bipartite matching constraint: each candidate S2/S3 record can match at most one S1 record
    (awarded to the S1 record with the highest confidence score clearing threshold).
    """
    df_pairs = df_feats[["s1_id", "cand_id"]].copy()
    df_pairs["proba"] = pred_probas

    # Filter pairs above threshold
    df_above = df_pairs[df_pairs["proba"] >= threshold].sort_values("proba", ascending=False)

    assigned_s23 = set()
    matches_by_s1 = {}

    for _, row in df_above.iterrows():
        s1_id = row["s1_id"]
        cand_id = row["cand_id"]

        # Enforce S2/S3 -> <= 1 S1 assignment
        if cand_id in assigned_s23:
            continue

        assigned_s23.add(cand_id)
        if s1_id not in matches_by_s1:
            matches_by_s1[s1_id] = set()
        matches_by_s1[s1_id].add(cand_id)

    return matches_by_s1


def export_matching_results(
    matches_by_s1: Dict[str, Set[str]],
    df_s1: pd.DataFrame,
    out_path: str
):
    """
    Export final entity resolution predictions to matching_results.tsv in exact tab-separated format:
    Header: source1_entity_id \t matched_entity_ids
    Where matched_entity_ids is comma-separated S2/S3 IDs (no spaces, empty if no match).
    Every S1 entity in test/train must have exactly one row.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = []
    for s1_id in df_s1["entity_id"].values:
        matched_ids = matches_by_s1.get(s1_id, set())
        match_str = ",".join(sorted(matched_ids))
        rows.append(f"{s1_id}\t{match_str}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("\n".join(rows) + "\n")

    print(f"[Matching Export] Wrote {len(rows)} matching results rows to {out_path}")
