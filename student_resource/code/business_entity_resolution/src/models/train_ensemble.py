"""
Model Training & Ensemble Module for Business Entity Resolution (Improved).

IMPROVEMENTS over v1:
- Expanded FEATURE_COLS to include all new features from improved interaction_features.py
- LightGBM: more trees (500), better regularization (min_child_samples, reg_alpha/lambda)
- CatBoost: more iterations (500), better depth, border_count tuning
- Added XGBoost as third ensemble member for diversity
- Weighted blend: LGB 40% + Cat 40% + XGB 20%
- Early stopping on validation set for LightGBM (prevents overfitting)
- GPU-aware fallback (if GPU not available, gracefully falls back to CPU)
"""

from typing import List
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


FEATURE_COLS = [
    # Name string features
    "name_lev", "name_sort", "name_set", "name_char3",
    "name_idf_jaccard", "name_len_ratio", "name_ntoken_ratio",
    # Core name features
    "core_lev", "core_sort", "core_set", "core_sorted_sim",
    "core_idf", "core_len_ratio",
    # Address features
    "addr_lev", "addr_set", "addr_char3", "addr_idf_jaccard", "addr_token_match",
    # Transliteration
    "translit_sim",
    # Structured features
    "country_match", "country_mismatch",
    "pincode_exact_match", "pincode_prefix_match",
    "both_has_pincode", "one_missing_pincode", "street_num_match",
    "suffix_match", "suffix_mismatch", "both_has_address",
    "has_landmark_s1", "has_landmark_s23",
    # Interaction features
    "name_addr_prod", "name_idf_addr_prod",
    "high_name_pin_mismatch", "name_match_addr_mismatch",
    # Rank / neighbourhood features
    "composite_sim", "cand_rank", "max_sim_for_s1", "min_sim_for_s1",
    "mean_sim_for_s1", "cand_count_for_s1",
    "sim_gap_to_best", "sim_percentile", "sim_gap_norm",
]


class EntityResolutionEnsemble:
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.lgb_model = None
        self.cat_model = None
        self.xgb_model = None
        self._available_features = None

    def _get_feature_cols(self, X: pd.DataFrame) -> List[str]:
        """Return only the feature columns that exist in X (handles old/new schema)."""
        return [c for c in FEATURE_COLS if c in X.columns]

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val: pd.DataFrame = None, y_val: np.ndarray = None):
        feat_cols = self._get_feature_cols(X)
        self._available_features = feat_cols

        pos_count = int(np.sum(y == 1))
        neg_count = int(np.sum(y == 0))
        scale_pos_weight = neg_count / max(pos_count, 1)
        print(f"[Model Fit] Training samples: {len(y)} | Positives: {pos_count} | Negatives: {neg_count} | SPW: {scale_pos_weight:.2f}")

        X_tr = X[feat_cols].values
        X_v = X_val[feat_cols].values if X_val is not None else None

        # ── 1. LightGBM ──────────────────────────────────────────────────────
        lgb_device = "gpu" if self.use_gpu else "cpu"
        lgb_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 63,
            "max_depth": 8,
            "min_child_samples": 50,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
            "subsample_freq": 1,
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "verbose": -1,
            "n_estimators": 500,
            "device": lgb_device,
        }

        print("[LightGBM] Training...")
        try:
            self.lgb_model = lgb.LGBMClassifier(**lgb_params)
            if X_val is not None and y_val is not None:
                self.lgb_model.fit(
                    X_tr, y,
                    eval_set=[(X_v, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
                )
            else:
                self.lgb_model.fit(X_tr, y)
            print(f"[LightGBM] Best iteration: {self.lgb_model.best_iteration_}")
        except Exception as e:
            print(f"[LightGBM] GPU failed ({e}), retrying on CPU...")
            lgb_params["device"] = "cpu"
            self.lgb_model = lgb.LGBMClassifier(**lgb_params)
            self.lgb_model.fit(X_tr, y)

        # ── 2. CatBoost ───────────────────────────────────────────────────────
        if HAS_CATBOOST:
            print("[CatBoost] Training...")
            cat_params = {
                "iterations": 500,
                "learning_rate": 0.03,
                "depth": 7,
                "l2_leaf_reg": 3.0,
                "border_count": 128,
                "scale_pos_weight": scale_pos_weight,
                "random_seed": 42,
                "verbose": 100,
                "early_stopping_rounds": 50 if X_val is not None else None,
            }
            if self.use_gpu:
                cat_params["task_type"] = "GPU"
            try:
                self.cat_model = CatBoostClassifier(**cat_params)
                if X_val is not None and y_val is not None:
                    self.cat_model.fit(X_tr, y, eval_set=(X_v, y_val))
                else:
                    self.cat_model.fit(X_tr, y)
            except Exception as e:
                print(f"[CatBoost] GPU failed ({e}), retrying on CPU...")
                cat_params.pop("task_type", None)
                self.cat_model = CatBoostClassifier(**cat_params)
                self.cat_model.fit(X_tr, y)
        else:
            print("[CatBoost] Not installed — skipping.")

        # ── 3. XGBoost ───────────────────────────────────────────────────────
        if HAS_XGB:
            print("[XGBoost] Training...")
            xgb_tree_method = "hist"
            xgb_device = "cuda" if self.use_gpu else "cpu"
            xgb_params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "learning_rate": 0.03,
                "max_depth": 7,
                "min_child_weight": 50,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "scale_pos_weight": scale_pos_weight,
                "random_state": 42,
                "n_estimators": 500,
                "tree_method": xgb_tree_method,
                "device": xgb_device,
                "verbosity": 0,
            }
            try:
                self.xgb_model = xgb.XGBClassifier(**xgb_params)
                if X_val is not None and y_val is not None:
                    self.xgb_model.fit(
                        X_tr, y,
                        eval_set=[(X_v, y_val)],
                        verbose=100,
                        early_stopping_rounds=50,
                    )
                else:
                    self.xgb_model.fit(X_tr, y)
            except Exception as e:
                print(f"[XGBoost] CUDA failed ({e}), retrying on CPU...")
                xgb_params["device"] = "cpu"
                self.xgb_model = xgb.XGBClassifier(**xgb_params)
                self.xgb_model.fit(X_tr, y)
        else:
            print("[XGBoost] Not installed — skipping.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        feat_cols = self._available_features or self._get_feature_cols(X)
        X_arr = X[feat_cols].values

        probas = []
        weights = []

        if self.lgb_model is not None:
            probas.append(self.lgb_model.predict_proba(X_arr)[:, 1])
            weights.append(0.40)

        if HAS_CATBOOST and self.cat_model is not None:
            probas.append(self.cat_model.predict_proba(X_arr)[:, 1])
            weights.append(0.40)

        if HAS_XGB and self.xgb_model is not None:
            probas.append(self.xgb_model.predict_proba(X_arr)[:, 1])
            weights.append(0.20)

        if not probas:
            raise ValueError("No models fitted.")

        # Normalize weights to sum to 1
        total_w = sum(weights)
        normalized = [w / total_w for w in weights]

        blended = sum(p * w for p, w in zip(probas, normalized))
        return blended
