"""
Master Pipeline Orchestrator for Business Entity Resolution (Amazon ML Challenge 2026).
Runs end-to-end: data ingestion → normalization → candidate blocking → feature extraction
→ ensemble model training (with early stopping) → threshold optimization
→ global assignment → submission validation.

IMPROVEMENTS over v1:
- Updated to use expanded blocking (7 keys), improved features (40+ columns), 3-model ensemble
- Passes X_val/y_val to ensemble for early stopping
- Creates output/ directory automatically
- Prints elapsed time per step
- Samples negatives for training (avoids extreme class imbalance at >1M scale)
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd

# Add parent of src/ to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.data.loaders import load_all_sources, load_ground_truth
from src.data.normalize import normalize_dataframe
from src.data.transliteration import build_transliteration_table
from src.blocking.rule_blocking import build_blocking_indices, generate_rule_candidates
from src.blocking.bm25_blocking import retrieve_tfidf_candidates
from src.blocking.recall_eval import evaluate_blocking_recall, export_candidate_pairs
from src.features.idf_jaccard import IDFEstimator
from src.features.interaction_features import build_candidate_feature_matrix
from src.models.train_ensemble import EntityResolutionEnsemble
from src.postprocessing.adaptive_threshold import find_optimal_threshold, evaluate_macro_f05
from src.postprocessing.global_assignment import resolve_global_assignments, export_matching_results


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s/60:.1f}m" if s >= 60 else f"{s:.1f}s"


def run_pipeline(
    data_dir: str = "dataset",
    output_dir: str = "output",
    sample_train_size: int = 150000,
    use_gpu: bool = True,
    neg_sample_ratio: float = 10.0,   # max negatives = ratio * positives
):
    t_global = time.time()
    print("=" * 70)
    print("   AMAZON ML CHALLENGE 2026 — BUSINESS ENTITY RESOLUTION PIPELINE")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — LOAD DATA
    # ─────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    print("\n[Step 1/7] Loading datasets...")
    train_dir = os.path.join(data_dir, "train")
    test_dir  = os.path.join(data_dir, "test")

    df_train_s1, df_train_s2, df_train_s3 = load_all_sources(train_dir, prefix="train")
    df_train_gt = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))
    df_test_s1, df_test_s2, df_test_s3    = load_all_sources(test_dir,  prefix="test")

    print(f"  Train  S1={len(df_train_s1):,} | S2={len(df_train_s2):,} | S3={len(df_train_s3):,} | GT={len(df_train_gt):,}")
    print(f"  Test   S1={len(df_test_s1):,}  | S2={len(df_test_s2):,}  | S3={len(df_test_s3):,}")
    print(f"  [{_elapsed(t0)}]")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — NORMALIZE
    # ─────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    print("\n[Step 2/7] Normalizing text & parsing addresses...")
    df_train_s1 = normalize_dataframe(df_train_s1)
    df_train_s2 = normalize_dataframe(df_train_s2)
    df_train_s3 = normalize_dataframe(df_train_s3)
    df_test_s1  = normalize_dataframe(df_test_s1)
    df_test_s2  = normalize_dataframe(df_test_s2)
    df_test_s3  = normalize_dataframe(df_test_s3)

    print("  Mining Indian transliteration patterns...")
    translit_table = build_transliteration_table(df_train_s1, df_train_s2, df_train_s3, df_train_gt)
    print(f"  [{_elapsed(t0)}]")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — IDF ESTIMATOR
    # ─────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    print("\n[Step 3/7] Fitting IDF Estimator over combined corpus...")
    idf_estimator = IDFEstimator()
    all_text = pd.concat([
        df_train_s1["name_norm"], df_train_s2["name_norm"], df_train_s3["name_norm"],
        df_test_s1["name_norm"],  df_test_s2["name_norm"],  df_test_s3["name_norm"],
    ])
    idf_estimator.fit(all_text)
    print(f"  IDF vocab size: {len(idf_estimator.idf_dict):,} tokens  [{_elapsed(t0)}]")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4 — CANDIDATE GENERATION FOR TEST
    # ─────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    print("\n[Step 4/7] Generating candidate pairs for Test set...")
    (pincode_idx_te, prefix4_idx_te, prefix6_idx_te,
     token_idx_te, bigram_idx_te, addr_token_idx_te,
     pin_prefix_idx_te) = build_blocking_indices(df_test_s2, df_test_s3)

    rule_cands_te = generate_rule_candidates(
        df_test_s1,
        pincode_idx_te, prefix4_idx_te, prefix6_idx_te,
        token_idx_te, bigram_idx_te, addr_token_idx_te, pin_prefix_idx_te,
    )
    tfidf_cands_te = retrieve_tfidf_candidates(df_test_s1, df_test_s2, df_test_s3, top_k=20)

    test_candidates = {}
    for s1_id in df_test_s1["entity_id"].values:
        test_candidates[s1_id] = (
            rule_cands_te.get(s1_id, set()) | tfidf_cands_te.get(s1_id, set())
        )

    n_test_pairs = sum(len(v) for v in test_candidates.values())
    print(f"  Test candidate pairs: {n_test_pairs:,}  [{_elapsed(t0)}]")

    cand_pairs_path = os.path.join(output_dir, "candidate_pairs.tsv")
    export_candidate_pairs(test_candidates, df_test_s1, cand_pairs_path)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5 — TRAINING CANDIDATES & FEATURE MATRIX
    # ─────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    print("\n[Step 5/7] Preparing training & validation features...")

    if sample_train_size and sample_train_size < len(df_train_s1):
        print(f"  Sampling {sample_train_size:,} S1 entities for training...")
        train_s1_sample = df_train_s1.sample(n=sample_train_size, random_state=42).reset_index(drop=True)
    else:
        train_s1_sample = df_train_s1

    (pincode_idx_tr, prefix4_idx_tr, prefix6_idx_tr,
     token_idx_tr, bigram_idx_tr, addr_token_idx_tr,
     pin_prefix_idx_tr) = build_blocking_indices(df_train_s2, df_train_s3)

    rule_cands_tr = generate_rule_candidates(
        train_s1_sample,
        pincode_idx_tr, prefix4_idx_tr, prefix6_idx_tr,
        token_idx_tr, bigram_idx_tr, addr_token_idx_tr, pin_prefix_idx_tr,
    )
    tfidf_cands_tr = retrieve_tfidf_candidates(train_s1_sample, df_train_s2, df_train_s3, top_k=20)

    train_candidates = {}
    for s1_id in train_s1_sample["entity_id"].values:
        train_candidates[s1_id] = (
            rule_cands_tr.get(s1_id, set()) | tfidf_cands_tr.get(s1_id, set())
        )

    # Evaluate recall ceiling on training sample
    gt_sample = df_train_gt[df_train_gt["source1_entity_id"].isin(train_candidates.keys())]
    evaluate_blocking_recall(train_candidates, gt_sample)

    # Build training pairs
    gt_map = df_train_gt.set_index("source1_entity_id")["target_set"].to_dict()
    train_pairs_list, train_labels = [], []

    for s1_id, cand_set in train_candidates.items():
        true_set = gt_map.get(s1_id, set())
        all_cands = cand_set | true_set  # always include positives

        positives = [(s1_id, cid) for cid in all_cands if cid in true_set]
        negatives = [(s1_id, cid) for cid in all_cands if cid not in true_set]

        train_pairs_list.extend(positives)
        train_labels.extend([1] * len(positives))

        # Downsample negatives to control class imbalance
        max_neg = int(len(positives) * neg_sample_ratio) if positives else min(len(negatives), 5)
        if len(negatives) > max_neg:
            import random
            rng = random.Random(42)
            negatives = rng.sample(negatives, max_neg)
        train_pairs_list.extend(negatives)
        train_labels.extend([0] * len(negatives))

    train_labels = np.array(train_labels)
    print(f"  Train pairs: {len(train_pairs_list):,} (pos={int(np.sum(train_labels)):,} neg={int(np.sum(1-train_labels)):,})  [{_elapsed(t0)}]")

    t0 = time.time()
    print("  Building training feature matrix...")
    df_all_feats = build_candidate_feature_matrix(
        train_pairs_list, train_s1_sample, df_train_s2, df_train_s3, idf_estimator, translit_table
    )
    print(f"  Feature matrix: {df_all_feats.shape}  [{_elapsed(t0)}]")

    # Validation split: 20% of training S1 entities
    val_s1_ids = set(train_s1_sample.sample(frac=0.2, random_state=42)["entity_id"])
    val_mask = df_all_feats["s1_id"].isin(val_s1_ids).values

    df_val_feats = df_all_feats[val_mask].reset_index(drop=True)
    y_val        = train_labels[val_mask]
    df_tr_feats  = df_all_feats[~val_mask].reset_index(drop=True)
    y_tr         = train_labels[~val_mask]

    print(f"  Train split: {len(df_tr_feats):,} | Val split: {len(df_val_feats):,}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6 — MODEL TRAINING & THRESHOLD
    # ─────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    print("\n[Step 6/7] Training ensemble (LightGBM + CatBoost + XGBoost)...")
    ensemble = EntityResolutionEnsemble(use_gpu=use_gpu)
    ensemble.fit(df_tr_feats, y_tr, X_val=df_val_feats, y_val=y_val)
    print(f"  Ensemble trained  [{_elapsed(t0)}]")

    print("  Tuning decision threshold on validation set...")
    val_probas = ensemble.predict_proba(df_val_feats)

    # Build a mini GT for validation entities only
    val_gt = df_train_gt[df_train_gt["source1_entity_id"].isin(val_s1_ids)]
    best_thresh, val_f05 = find_optimal_threshold(df_val_feats, val_probas, val_gt)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7 — TEST INFERENCE & SUBMISSION
    # ─────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    print("\n[Step 7/7] Generating final predictions for Test set...")

    test_pairs_list = [
        (s1_id, cid)
        for s1_id, cand_set in test_candidates.items()
        for cid in cand_set
    ]
    print(f"  Building feature matrix for {len(test_pairs_list):,} test pairs...")

    df_test_feats = build_candidate_feature_matrix(
        test_pairs_list, df_test_s1, df_test_s2, df_test_s3, idf_estimator, translit_table
    )

    if len(df_test_feats) > 0:
        test_probas = ensemble.predict_proba(df_test_feats)
        matches_by_s1 = resolve_global_assignments(df_test_feats, test_probas, threshold=best_thresh)
    else:
        matches_by_s1 = {}

    matching_path = os.path.join(output_dir, "matching_results.tsv")
    export_matching_results(matches_by_s1, df_test_s1, matching_path)

    matched = sum(1 for v in matches_by_s1.values() if v)
    singletons = len(df_test_s1) - matched
    print(f"  Matched entities: {matched:,} | Singletons predicted: {singletons:,}  [{_elapsed(t0)}]")

    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETE! Total time: {_elapsed(t_global)}")
    print(f"  Outputs: {output_dir}/matching_results.tsv")
    print(f"           {output_dir}/candidate_pairs.tsv")
    print("=" * 70)
    print(f"\n  Validate with:")
    print(f"  python utils/validate_submission.py \\")
    print(f"      --matching {output_dir}/matching_results.tsv \\")
    print(f"      --candidate {output_dir}/candidate_pairs.tsv \\")
    print(f"      --test-dir {test_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument("--data-dir",       type=str,   default="dataset",  help="Dataset root directory")
    parser.add_argument("--output-dir",     type=str,   default="output",   help="Output directory")
    parser.add_argument("--sample-train",   type=int,   default=150000,     help="Train S1 sample size (0 = all)")
    parser.add_argument("--neg-ratio",      type=float, default=10.0,       help="Neg/pos ratio for training pairs")
    parser.add_argument("--no-gpu",         action="store_true",            help="Disable GPU acceleration")
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sample_train_size=args.sample_train,
        use_gpu=not args.no_gpu,
        neg_sample_ratio=args.neg_ratio,
    )
