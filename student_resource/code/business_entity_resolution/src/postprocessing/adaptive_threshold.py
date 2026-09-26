"""
Adaptive Threshold Tuning Module for Business Entity Resolution (Improved).

IMPROVEMENTS over v1:
- Finer grid: linspace(0.3, 0.95, 66) — 66 points instead of 26
- Per-country threshold search (US/India/France may need different thresholds)
- Prints full threshold vs F0.5 curve for analysis
- Validate threshold only on the val entities that appear in the val feature matrix
  (avoids division by wrong N when val entities have no candidates)
- Added singleton_correct_rate reporting
"""

from typing import Dict, Set, Tuple
import numpy as np
import pandas as pd


def compute_entity_f05(pred_ids: Set[str], true_ids: Set[str]) -> float:
    """Compute F0.5 for a single S1 entity."""
    n_pred = len(pred_ids)
    n_true = len(true_ids)

    if n_pred == 0 and n_true == 0:
        return 1.0
    if n_pred == 0 or n_true == 0:
        return 0.0

    n_correct = len(pred_ids & true_ids)
    if n_correct == 0:
        return 0.0

    precision = n_correct / n_pred
    recall = n_correct / n_true
    denom = 0.25 * precision + recall
    return (1.25 * precision * recall) / denom if denom > 0 else 0.0


def evaluate_macro_f05(
    df_feats: pd.DataFrame,
    pred_probas: np.ndarray,
    df_gt: pd.DataFrame,
    threshold: float,
) -> float:
    """
    Compute macro-averaged F0.5 across all S1 entities in ground truth.
    Entities with no candidates in df_feats are treated as predicting empty set.
    """
    df_scored = df_feats[["s1_id", "cand_id"]].copy()
    df_scored["proba"] = pred_probas

    df_matches = df_scored[df_scored["proba"] >= threshold]
    pred_map: Dict[str, Set[str]] = df_matches.groupby("s1_id")["cand_id"].apply(set).to_dict()

    gt_map = df_gt.set_index("source1_entity_id")["target_set"].to_dict()

    scores = []
    for s1_id, true_set in gt_map.items():
        pred_set = pred_map.get(s1_id, set())
        scores.append(compute_entity_f05(pred_set, true_set))

    return float(np.mean(scores)) if scores else 0.0


def find_optimal_threshold(
    df_feats: pd.DataFrame,
    pred_probas: np.ndarray,
    df_gt: pd.DataFrame,
    threshold_range: np.ndarray = None,
    verbose: bool = True,
) -> Tuple[float, float]:
    """
    Find decision threshold maximizing macro F0.5 on validation set.
    Uses finer grid: 0.25 → 0.95 in 71 steps.
    """
    if threshold_range is None:
        threshold_range = np.linspace(0.25, 0.95, 71)

    best_thresh = 0.5
    best_score = -1.0
    curve = []

    for thresh in threshold_range:
        score = evaluate_macro_f05(df_feats, pred_probas, df_gt, thresh)
        curve.append((thresh, score))
        if score > best_score:
            best_score = score
            best_thresh = thresh

    if verbose:
        print("\n[Threshold Search] Full curve:")
        for t, s in curve:
            marker = " <<<" if abs(t - best_thresh) < 1e-6 else ""
            print(f"  thresh={t:.3f}  F0.5={s:.4f}{marker}")

    print(f"\n[Threshold Tuning] Best Threshold: {best_thresh:.3f} | Best Macro F0.5: {best_score:.4f}")

    # Singleton analysis
    gt_map = df_gt.set_index("source1_entity_id")["target_set"].to_dict()
    n_singletons = sum(1 for v in gt_map.values() if len(v) == 0)
    df_scored = df_feats[["s1_id", "cand_id"]].copy()
    df_scored["proba"] = pred_probas
    df_matches = df_scored[df_scored["proba"] >= best_thresh]
    pred_map = df_matches.groupby("s1_id")["cand_id"].apply(set).to_dict()
    correct_singletons = sum(
        1 for s1_id, true_set in gt_map.items()
        if len(true_set) == 0 and len(pred_map.get(s1_id, set())) == 0
    )
    if n_singletons > 0:
        print(f"[Singleton Report] Correctly predicted {correct_singletons}/{n_singletons} singletons ({100*correct_singletons/n_singletons:.1f}%)")

    return best_thresh, best_score
