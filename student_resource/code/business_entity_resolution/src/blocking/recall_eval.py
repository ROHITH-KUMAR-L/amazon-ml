"""
Recall Evaluation & Candidate Export Module for Business Entity Resolution.
Combines rule blocking and TF-IDF candidates, measures recall ceiling, and outputs candidate_pairs.tsv.
"""

import os
from typing import Dict, Set
import pandas as pd


def evaluate_blocking_recall(candidates_by_s1: Dict[str, Set[str]], df_gt: pd.DataFrame) -> dict:
    """
    Compute candidate generation recall ceiling and reduction ratio on ground truth.
    Recall = (Total true matches captured in candidate set) / (Total true matches in GT)
    """
    total_true_matches = 0
    captured_true_matches = 0
    total_candidates = 0
    
    # Ground truth map: s1_id -> target_set
    gt_map = df_gt.set_index("source1_entity_id")["target_set"].to_dict()
    
    for s1_id, target_set in gt_map.items():
        n_true = len(target_set)
        total_true_matches += n_true
        
        cands = candidates_by_s1.get(s1_id, set())
        total_candidates += len(cands)
        
        if n_true > 0:
            captured_true_matches += len(target_set.intersection(cands))
            
    recall = captured_true_matches / total_true_matches if total_true_matches > 0 else 1.0
    avg_cands_per_entity = total_candidates / len(gt_map) if len(gt_map) > 0 else 0.0
    
    print(f"[Blocking Eval] Recall Ceiling: {recall:.4f} ({captured_true_matches}/{total_true_matches} matches captured)")
    print(f"[Blocking Eval] Avg Candidates/Entity: {avg_cands_per_entity:.2f}")
    
    return {
        "recall": recall,
        "captured": captured_true_matches,
        "total_true": total_true_matches,
        "avg_candidates": avg_cands_per_entity
    }


def export_candidate_pairs(candidates_by_s1: Dict[str, Set[str]], df_s1: pd.DataFrame, out_path: str):
    """
    Export candidate pairs in the exact format required for candidate_pairs.tsv:
    Header: source1_entity_id \t candidate_entity_ids
    Where candidate_entity_ids is comma-separated S2/S3 entity_ids (no spaces).
    Every S1 entity in test/train must have exactly one row.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    rows = []
    for s1_id in df_s1["entity_id"].values:
        cands = candidates_by_s1.get(s1_id, set())
        cand_str = ",".join(sorted(cands))
        rows.append(f"{s1_id}\t{cand_str}")
        
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        f.write("\n".join(rows) + "\n")
        
    print(f"[Candidate Export] Wrote {len(rows)} candidate rows to {out_path}")
