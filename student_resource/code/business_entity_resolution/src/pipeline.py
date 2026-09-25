"""
Master Pipeline Orchestrator for Business Entity Resolution (Amazon ML Challenge 2026).
Runs end-to-end data ingestion, normalization, candidate blocking, feature extraction, ensemble model training,
threshold optimization, global assignment, and submission validation.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# Add src to python path
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


def run_pipeline(
    data_dir: str = "dataset",
    output_dir: str = "output",
    sample_train_size: int = 150000,
    use_gpu: bool = True
):
    print("=" * 70)
    print("      AMAZON ML CHALLENGE 2026 — BUSINESS ENTITY RESOLUTION PIPELINE      ")
    print("=" * 70)
    
    # ---------------------------------------------------------
    # 1. LOAD DATASETS
    # ---------------------------------------------------------
    print("\n[Step 1/7] Loading datasets...")
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    
    df_train_s1, df_train_s2, df_train_s3 = load_all_sources(train_dir, prefix="train")
    df_train_gt = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))
    
    df_test_s1, df_test_s2, df_test_s3 = load_all_sources(test_dir, prefix="test")
    
    print(f"Train S1: {len(df_train_s1)} | S2: {len(df_train_s2)} | S3: {len(df_train_s3)} | GT: {len(df_train_gt)}")
    print(f"Test S1: {len(df_test_s1)} | S2: {len(df_test_s2)} | S3: {len(df_test_s3)}")
    
    # ---------------------------------------------------------
    # 2. DATA NORMALIZATION
    # ---------------------------------------------------------
    print("\n[Step 2/7] Normalizing text & parsing addresses...")
    df_train_s1 = normalize_dataframe(df_train_s1)
    df_train_s2 = normalize_dataframe(df_train_s2)
    df_train_s3 = normalize_dataframe(df_train_s3)
    
    df_test_s1 = normalize_dataframe(df_test_s1)
    df_test_s2 = normalize_dataframe(df_test_s2)
    df_test_s3 = normalize_dataframe(df_test_s3)
    
    # Mine transliteration patterns from India training ground truth
    print("Mining Indian transliteration substitution patterns...")
    translit_table = build_transliteration_table(
        df_train_s1, df_train_s2, df_train_s3, df_train_gt
    )
    
    # ---------------------------------------------------------
    # 3. FIT IDF ESTIMATOR
    # ---------------------------------------------------------
    print("\n[Step 3/7] Fitting IDF Estimator over corpus...")
    idf_estimator = IDFEstimator()
    all_text = pd.concat([
        df_train_s1["name_norm"], df_train_s2["name_norm"], df_train_s3["name_norm"],
        df_test_s1["name_norm"], df_test_s2["name_norm"], df_test_s3["name_norm"]
    ])
    idf_estimator.fit(all_text)
    
    # ---------------------------------------------------------
    # 4. CANDIDATE GENERATION (BLOCKING) FOR TEST
    # ---------------------------------------------------------
    print("\n[Step 4/7] Generating candidate pairs for Test set...")
    pincode_idx, prefix_idx, token_idx = build_blocking_indices(df_test_s2, df_test_s3)
    rule_cands_test = generate_rule_candidates(df_test_s1, pincode_idx, prefix_idx, token_idx)
    tfidf_cands_test = retrieve_tfidf_candidates(df_test_s1, df_test_s2, df_test_s3, top_k=15)
    
    # Union candidates
    test_candidates = {}
    for s1_id in df_test_s1["entity_id"].values:
        c1 = rule_cands_test.get(s1_id, set())
        c2 = tfidf_cands_test.get(s1_id, set())
        test_candidates[s1_id] = c1.union(c2)
        
    cand_pairs_path = os.path.join(output_dir, "candidate_pairs.tsv")
    export_candidate_pairs(test_candidates, df_test_s1, cand_pairs_path)
    
    # ---------------------------------------------------------
    # 5. CANDIDATE GENERATION & TRAINING PREPARATION
    # ---------------------------------------------------------
    print("\n[Step 5/7] Preparing training & validation candidate sample...")
    if sample_train_size and sample_train_size < len(df_train_s1):
        print(f"Sampling {sample_train_size} S1 entities for training...")
        train_sample_s1 = df_train_s1.sample(n=sample_train_size, random_state=42).reset_index(drop=True)
    else:
        train_sample_s1 = df_train_s1
        
    pincode_idx_tr, prefix_idx_tr, token_idx_tr = build_blocking_indices(df_train_s2, df_train_s3)
    rule_cands_tr = generate_rule_candidates(train_sample_s1, pincode_idx_tr, prefix_idx_tr, token_idx_tr)
    tfidf_cands_tr = retrieve_tfidf_candidates(train_sample_s1, df_train_s2, df_train_s3, top_k=15)
    
    train_candidates = {}
    for s1_id in train_sample_s1["entity_id"].values:
        c1 = rule_cands_tr.get(s1_id, set())
        c2 = tfidf_cands_tr.get(s1_id, set())
        train_candidates[s1_id] = c1.union(c2)
        
    # Evaluate recall ceiling
    evaluate_blocking_recall(train_candidates, df_train_gt)
    
    # Build candidate pairs list & ground truth labels
    gt_map = df_train_gt.set_index("source1_entity_id")["target_set"].to_dict()
    
    train_pairs_list = []
    train_labels = []
    
    for s1_id, cand_set in train_candidates.items():
        true_set = gt_map.get(s1_id, set())
        
        # Add true matches explicitly if missed by blocking to train classifier on positive examples
        all_cands = cand_set.union(true_set)
        
        for cand_id in all_cands:
            train_pairs_list.append((s1_id, cand_id))
            train_labels.append(1 if cand_id in true_set else 0)
            
    train_labels = np.array(train_labels)
    print(f"Extracted {len(train_pairs_list)} training pair features (Positives: {np.sum(train_labels == 1)})...")
    
    df_train_feats = build_candidate_feature_matrix(
        train_pairs_list, train_sample_s1, df_train_s2, df_train_s3, idf_estimator, translit_table
    )
    
    # Validation split (20% of training entities)
    val_s1_ids = set(train_sample_s1.sample(frac=0.2, random_state=42)["entity_id"])
    val_mask = df_train_feats["s1_id"].isin(val_s1_ids)
    
    df_val_feats = df_train_feats[val_mask].reset_index(drop=True)
    y_val = train_labels[val_mask.values]
    
    df_tr_feats = df_train_feats[~val_mask].reset_index(drop=True)
    y_tr = train_labels[(~val_mask).values]
    
    # ---------------------------------------------------------
    # 6. MODEL TRAINING & THRESHOLD OPTIMIZATION
    # ---------------------------------------------------------
    print("\n[Step 6/7] Training GBDT Ensemble with GPU acceleration...")
    ensemble = EntityResolutionEnsemble(use_gpu=use_gpu)
    ensemble.fit(df_tr_feats, y_tr)
    
    print("Tuning decision threshold on validation fold...")
    val_probas = ensemble.predict_proba(df_val_feats)
    best_thresh, val_f05 = find_optimal_threshold(df_val_feats, val_probas, df_train_gt)
    
    # ---------------------------------------------------------
    # 7. INFERENCE & SUBMISSION GENERATION FOR TEST
    # ---------------------------------------------------------
    print("\n[Step 7/7] Generating final predictions for Test set...")
    test_pairs_list = []
    for s1_id, cand_set in test_candidates.items():
        for cand_id in cand_set:
            test_pairs_list.append((s1_id, cand_id))
            
    print(f"Building feature matrix for {len(test_pairs_list)} test candidate pairs...")
    df_test_feats = build_candidate_feature_matrix(
        test_pairs_list, df_test_s1, df_test_s2, df_test_s3, idf_estimator, translit_table
    )
    
    if len(df_test_feats) > 0:
        test_probas = ensemble.predict_proba(df_test_feats)
        matches_by_s1 = resolve_global_assignments(df_test_feats, test_probas, threshold=best_thresh)
    else:
        matches_by_s1 = {}
        
    matching_results_path = os.path.join(output_dir, "matching_results.tsv")
    export_matching_results(matches_by_s1, df_test_s1, matching_results_path)
    
    print("\n" + "=" * 70)
    print("  PIPELINE EXECUTION COMPLETE! OUTPUT FILES WRITTEN TO 'output/'  ")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Dataset root directory")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--sample-train", type=int, default=150000, help="Train sample size")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    
    args = parser.parse_args()
    
    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sample_train_size=args.sample_train,
        use_gpu=not args.no_gpu
    )
