"""
Adaptive Threshold Tuning Module for Business Entity Resolution.
Computes local Macro F0.5 score per entity and performs grid search for optimal decision threshold.
"""

from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd


def compute_entity_f05(pred_ids: Set[str], true_ids: Set[str]) -> float:
    """
    Compute F0.5 score for a single S1 entity given predicted match IDs and ground truth match IDs.
    """
    n_pred = len(pred_ids)
    n_true = len(true_ids)

    if n_pred == 0 and n_true == 0:
        return 1.0
    if n_pred == 0 or n_true == 0:
        return 0.0

    n_correct = len(pred_ids.intersection(true_ids))
    if n_correct == 0:
        return 0.0

    precision = n_correct / n_pred
    recall = n_correct / n_true

    denom = (0.25 * precision) + recall
    if denom == 0:
        return 0.0

    return (1.25 * precision * recall) / denom


def evaluate_macro_f05(
    df_feats: pd.DataFrame,
    pred_probas: np.ndarray,
    df_gt: pd.DataFrame,
    threshold: float
) -> float:
    """
    Compute macro-averaged F0.5 score across all S1 entities in ground truth for a given candidate score threshold.
    """
    df_scored = df_feats.copy()
    df_scored["proba"] = pred_probas

    # Filter pairs clearing the threshold
    df_matches = df_scored[df_scored["proba"] >= threshold]

    # Group predicted matches by s1_id
    pred_map = df_matches.groupby("s1_id")["cand_id"].apply(set).to_dict()
    gt_map = df_gt.set_index("source1_entity_id")["target_set"].to_dict()

    scores = []
    for s1_id, true_set in gt_map.items():
        pred_set = pred_map.get(s1_id, set())
        f05 = compute_entity_f05(pred_set, true_set)
        scores.append(f05)

    return float(np.mean(scores))


def find_optimal_threshold(
    df_feats: pd.DataFrame,
    pred_probas: np.ndarray,
    df_gt: pd.DataFrame,
    threshold_range: np.ndarray = np.linspace(0.4, 0.9, 26)
) -> Tuple[float, float]:
    """
    Find decision threshold that maximizes macro F0.5 score on validation set.
    """
    best_thresh = 0.5
    best_score = -1.0

    for thresh in threshold_range:
        score = evaluate_macro_f05(df_feats, pred_probas, df_gt, thresh)
        if score > best_score:
            best_score = score
            best_thresh = thresh

    print(f"[Threshold Tuning] Best Threshold: {best_thresh:.3f} | Best Macro F0.5: {best_score:.4f}")
    return best_thresh, best_score
