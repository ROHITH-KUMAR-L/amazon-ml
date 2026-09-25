# Implementation Plan — Business Entity Resolution (Amazon ML Challenge 2026)

This document outlines the technical plan for building an end-to-end Machine Learning solution for the **Business Entity Resolution Challenge**. Given business records from 3 independent data sources (`S1` reference source, `S2`, `S3`) with noisy fields (name, address, country), the goal is to identify all matching records in `S2` and `S3` for every `S1` entity, evaluated on **macro-averaged $F_{0.5}$ score**.

---

## Dataset Directory & Structural Inconsistencies Identified & Corrected

During initial exploration of the project workspace (`/home/rohith/Documents/AMAZON_ML`), several dataset and directory inconsistencies were identified and handled:

1. **macOS System Junk Files (`.DS_Store`)**:
   - **Issue**: OS junk files `student_resource/.DS_Store` and `student_resource/dataset/.DS_Store` were present.
   - **Action Taken**: Removed all `.DS_Store` files to ensure clean directory indexing and clean zip archives.

2. **Documentation File Mismatch**:
   - **Issue**: `student_resource/Documentation_template.md` was an empty 75-line template, whereas `Documentation_template_v2.md` in the root workspace contained a complete 720-line technical methodology specification (v2.0 architecture, feature engineering, adaptive thresholding, transliteration module, pseudo-labeling, etc.).
   - **Action Taken**: Synchronized `Documentation_template_v2.md` to `student_resource/Documentation_template.md` so that automated submission packaging correctly includes the complete methodology documentation.

3. **Ground Truth Row Order Mismatch**:
   - **Issue**: In `student_resource/dataset/train/`, `train_ground_truth.tsv` contains all 2,206,821 `source1_entity_id` values, but in a **shuffled order** relative to `train_source1.tsv`.
   - **Action Taken**: Pipeline loaders will strictly perform key-based merges on `source1_entity_id` / `entity_id` rather than relying on positional line order.

4. **Missing Values & Nulls in Address / Name Fields**:
   - **Issue**: `train_source2` (168k missing addresses), `train_source3` (175k missing addresses), `test_source2` (129k missing addresses), and `test_source3` (136k missing addresses) have significant missing fields, along with occasional missing `business_name` entries.
   - **Action Taken**: Data normalization module will map missing strings to empty strings `""` and introduce explicit binary missingness flags (`has_address`, `has_name`, `has_pincode`) so models distinguish absent fields from mismatched text.

5. **Open-Set Country Distribution (Domain Shift)**:
   - **Issue**: Training data contains only `US` and `India`, while test data introduces an unseen third country, `France` (~259k entities, ~15% of test data).
   - **Action Taken**: Normalization and feature extraction will treat `country` as an open-set string label using language-agnostic cleanups and language-independent features to ensure seamless execution on French entities.

6. **Submission Output & Code Directory Structure**:
   - **Issue**: Standard output folder `student_resource/output/` and code structure `student_resource/code/business_entity_resolution/` do not yet exist.
   - **Action Taken**: Will be automatically created during pipeline execution to comply with competition packaging standards.

---

## User Review Required

> [!IMPORTANT]
> **Key Decisions & Strategy Alignment**:
> 1. **Evaluation Metric ($F_{0.5}$)**: $F_{0.5}$ weights Precision 2× over Recall. Penalizing false merges (wrong matches) is twice as severe as missing a match. Correctly predicting singletons (empty match list) awards full 1.0 credit per entity.
> 2. **Phased Model Scale**: We will implement a high-performance **Multi-Key Rule + BM25 Blocking + Engineered Features + GBDT Ensemble (LightGBM/CatBoost) + Bipartite Assignment** pipeline as the core architecture. If compute time permits, embedding bi-encoders / cross-encoders will be integrated as additional feature layers as outlined in `Documentation_template_v2.md`.

---

## Open Questions

> [!NOTE]
> Please let us know if you have specific constraints or preferences regarding:
> - Hardware/Compute preference: Do you prefer CPU-only GBDT training, or should GPU acceleration be enabled if available?
> - Validation sample size: Training dataset is large (~2.2M S1 records, ~10M S2/S3 records). We plan to run cross-validation on a representative cluster-aware stratified sample (e.g., 200k S1 entities) for rapid iteration before full-dataset fit.

---

## Proposed Changes

We will create a modular Python package under `student_resource/code/business_entity_resolution/` and execute the pipeline to generate outputs in `student_resource/output/`.

---

### Component 1: Data Preprocessing & Pre-flight Validation
#### [NEW] [loaders.py](file:///home/rohith/Documents/AMAZON_ML/student_resource/code/business_entity_resolution/src/data/loaders.py)
- Safe TSV reader with explicit tab separation (`sep="\t"`), malformed line handling, missing value imputation, and key-based GT joining.

#### [NEW] [normalize.py](file:///home/rohith/Documents/AMAZON_ML/student_resource/code/business_entity_resolution/src/data/normalize.py)
- Unicode NFKC normalization, lowercase conversion, legal suffix standardization (`corp`, `inc`, `pvt`, `ltd`), address regex parsing (extracting PIN codes, street numbers, cities, landmarks), and binary missingness indicators (`has_pincode`, `has_address`).

#### [NEW] [transliteration.py](file:///home/rohith/Documents/AMAZON_ML/student_resource/code/business_entity_resolution/src/data/transliteration.py)
- Mined character substitution table from `India` true matches in `train_ground_truth.tsv` (e.g., `v<->w`, `s<->sh`, `i<->ee`), creating a custom weighted edit distance feature for Indian entity names.

---

### Component 2: Candidate Generation (Blocking)
#### [NEW] [rule_blocking.py](file:///home/rohith/Documents/AMAZON_ML/student_resource/code/business_entity_resolution/src/blocking/rule_blocking.py)
- Multi-key rule blocking (union of: `same_country + same_pincode`, `same_country + matching_city_tokens`, `Metaphone(sorted_name_tokens)`, `char_3gram_minhash`).

#### [NEW] [bm25_blocking.py](file:///home/rohith/Documents/AMAZON_ML/student_resource/code/business_entity_resolution/src/blocking/bm25_blocking.py)
- Fast BM25 index over character 3-grams of concatenated normalized `name + address` to retrieve top-K candidates for each `S1` entity.

#### [NEW] [recall_eval.py](file:///home/rohith/Documents/AMAZON_ML/student_resource/code/business_entity_resolution/src/blocking/recall_eval.py)
- Evaluates candidate recall ceiling (>95% target) and reduction ratio (>99.5% target) on validation set. Outputs `output/candidate_pairs.tsv`.

---

### Component 3: Feature Engineering
#### [NEW] [string_features.py](file:///home/rohith/Documents/AMAZON_ML/student_resource/code/business_entity_resolution/src/features/string_features.py)
- Levenshtein ratio, Jaro-Winkler, Token-Sort ratio, Token-Set ratio, character n-gram cosine similarity computed separately for names and addresses.

#### [NEW] [idf_jaccard.py](file:///home/rohith/Documents/AMAZON_ML/student_resource/code/business_entity_resolution/src/features/idf_jaccard.py)
- Inverse Document Frequency (IDF) weighted token Jaccard similarity, ensuring rare brand/family names carry higher weight than generic words (`pvt`, `store`, `road`).

#### [NEW] [structured_features.py](file:///home/rohith/Documents/AMAZON_ML/student_resource/code/business_entity_resolution/src/features/structured_features.py)
- PIN code exact/prefix match, street number match, city/state token overlap, country match indicator.

#### [NEW] [interaction_features.py](file:///home/rohith/Documents/AMAZON_ML/student_resource/code/business_entity_resolution/src/features/interaction_features.py)
- Cross-field interaction terms: `name_sim * address_sim`, `name_sim * country_match`, `high_name_sim AND pin_mismatch` flag, `neighborhood_rank`, and `score_gap_to_second_best`.

---

### Component 4: Model Training, Thresholding & Bipartite Post-processing
#### [NEW] [train_ensemble.py](file:///home/rohith/Documents/AMAZON_ML/student_resource/code/business_entity_resolution/src/models/train_ensemble.py)
- Trains diverse GBDT models (Full-feature LightGBM, Name-heavy CatBoost/LightGBM, Address-heavy LightGBM) with class imbalance weighting (`scale_pos_weight`).

#### [NEW] [adaptive_threshold.py](file:///home/rohith/Documents/AMAZON_ML/student_resource/code/business_entity_resolution/src/postprocessing/adaptive_threshold.py)
- Exact macro-$F_{0.5}$ grid search per entity context (candidate density, missingness bucket, country), enforcing explicit singleton threshold cutoffs.

#### [NEW] [global_assignment.py](file:///home/rohith/Documents/AMAZON_ML/student_resource/code/business_entity_resolution/src/postprocessing/global_assignment.py)
- Enforces 1-to-many constraint ($S2/S3 \to \le 1\ S1$) via greedy max-weight bipartite matching, resolving contested entities and outputting `output/matching_results.tsv`.

---

### Component 5: Validation & Packaging
#### [NEW] [pipeline.py](file:///home/rohith/Documents/AMAZON_ML/student_resource/code/business_entity_resolution/src/pipeline.py)
- End-to-end master runner executing normalizing -> blocking -> feature extraction -> inference -> postprocessing -> submission validation.

#### [NEW] [package_submission.py](file:///home/rohith/Documents/AMAZON_ML/student_resource/code/business_entity_resolution/package_submission.py)
- Packs final outputs, source code, requirements, and methodology documentation into `<team_name>_submission.zip`.

---

## Verification Plan

### Automated Tests
1. **Format & Constraint Validation**:
   - Run `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids` from `student_resource/`.
   - Verify exit code `0` (`PASS`) with 0 errors.

2. **Offline Leaderboard Macro-$F_{0.5}$ Evaluation**:
   - Evaluate cluster-aware validation fold using exact macro-$F_{0.5}$ formula.
   - Verify prediction compliance: 1 row per test S1 entity, no S1 self-matches, no duplicate IDs within match list, all IDs present in test set.

### Manual Verification
- Verify zip package contents: `output/matching_results.tsv`, `output/candidate_pairs.tsv`, `code/business_entity_resolution/src/`, `code/business_entity_resolution/README.md`, `code/business_entity_resolution/requirements.txt`, `Documentation_template.md`.
