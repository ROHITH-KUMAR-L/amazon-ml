# Business Entity Resolution — Reproduction Guide

This package contains the self-contained source code for the **Amazon ML Challenge 2026 — Business Entity Resolution** challenge solution.

## Quick Reproduction Guide

### 1. Environment Setup
Install dependencies in your Python virtual environment:

```bash
pip install -r requirements.txt
```

### 2. Run End-to-End Pipeline
Execute the full pipeline from the `student_resource/` directory:

```bash
PYTHONPATH=code/business_entity_resolution python3 code/business_entity_resolution/src/pipeline.py \
    --data-dir dataset \
    --output-dir output
```

This single command performs:
1. Data loading and key-based ground truth alignment.
2. Text canonicalization (NFKC, lowercasing, legal suffix normalization) and address regex parsing.
3. Mining Indian transliteration character substitution patterns from training true matches.
4. Candidate generation via rule blocking & country-partitioned BM25 char 3-gram indexing.
5. Exporting `output/candidate_pairs.tsv`.
6. IDF estimator fitting and feature extraction (string metrics, structured address match, missingness flags, cross-field interactions, candidate ranks).
7. GBDT Ensemble training (LightGBM + CatBoost) with GPU acceleration and class-imbalance weighting.
8. Decision threshold optimization targeting macro $F_{0.5}$.
9. Greedy bipartite 1-to-many global consistency post-processing ($S2/S3 \to \le 1\ S1$).
10. Exporting `output/matching_results.tsv`.

### 3. Submission Format Validation
Run the validator script to verify local compliance:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
