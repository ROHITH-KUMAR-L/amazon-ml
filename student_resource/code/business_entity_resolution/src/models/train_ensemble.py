"""
Model Training & Ensemble Module for Business Entity Resolution.
Trains LightGBM / CatBoost models with class imbalance weighting and GPU acceleration.
"""

from typing import List, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


FEATURE_COLS = [
    "name_lev", "name_sort", "name_set", "core_lev",
    "addr_lev", "addr_sort", "addr_set", "char3_sim",
    "name_idf_jaccard", "addr_idf_jaccard", "translit_sim",
    "name_addr_prod", "name_country_prod", "high_name_pin_mismatch",
    "country_match", "pincode_exact_match", "pincode_prefix_match",
    "both_has_pincode", "one_missing_pincode", "street_num_match",
    "suffix_match", "suffix_mismatch", "both_has_address",
    "has_landmark_s1", "has_landmark_s23",
    "composite_sim", "cand_rank", "max_sim_for_s1", "cand_count_for_s1", "sim_gap_to_best"
]


class EntityResolutionEnsemble:
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.lgb_model = None
        self.cat_model = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        pos_count = np.sum(y == 1)
        neg_count = np.sum(y == 0)
        scale_pos_weight = neg_count / max(pos_count, 1)
        print(f"[Model Fit] Training samples: {len(y)} (Positives: {pos_count}, Negatives: {neg_count}, Scale Weight: {scale_pos_weight:.2f})")

        # 1. Train LightGBM
        lgb_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 7,
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "verbose": -1,
            "n_estimators": 250
        }
        if self.use_gpu:
            lgb_params["device"] = "gpu"

        self.lgb_model = lgb.LGBMClassifier(**lgb_params)
        self.lgb_model.fit(X[FEATURE_COLS], y)

        # 2. Train CatBoost
        cat_params = {
            "iterations": 250,
            "learning_rate": 0.05,
            "depth": 6,
            "scale_pos_weight": scale_pos_weight,
            "random_seed": 42,
            "verbose": False
        }
        if self.use_gpu:
            cat_params["task_type"] = "GPU"

        self.cat_model = CatBoostClassifier(**cat_params)
        self.cat_model.fit(X[FEATURE_COLS], y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.lgb_model is None or self.cat_model is None:
            raise ValueError("Ensemble model is not fitted yet.")

        p_lgb = self.lgb_model.predict_proba(X[FEATURE_COLS])[:, 1]
        p_cat = self.cat_model.predict_proba(X[FEATURE_COLS])[:, 1]

        # Equal weighted blend
        return 0.5 * p_lgb + 0.5 * p_cat
