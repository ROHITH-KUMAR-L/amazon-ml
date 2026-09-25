# Amazon ML Challenge 2026 — Progress & Execution Task Status

**Project:** Business Entity Resolution (ER)  
**Status Date:** September 25, 2026  

---

## 1. Currently Running Command / Activity

I was running the master orchestrator script (`pipeline.py`) in the background:
```bash
PYTHONPATH=code/business_entity_resolution /home/rohith/.venvs/myenv/bin/python code/business_entity_resolution/src/pipeline.py \
    --data-dir dataset \
    --output-dir output \
    --sample-train 100000
```

### Execution Steps Completed during the Run:
1. **[Step 1/7] Data Ingestion**: Loaded 2,206,821 $S1$ train entities, 5,034,616 $S2$ train entities, 5,285,603 $S3$ train entities, 2,206,821 ground truth mappings, 1,732,544 $S1$ test entities, 4,887,273 $S2$ test entities, and 5,082,316 $S3$ test entities.
2. **[Step 2/7] Text Canonicalization & Regex Address Parsing**: Applied vectorized NFKC normalization, legal suffix extraction (`corp`, `inc`, `pvt`, `ltd`), address regex parsing (PIN codes, street numbers, landmark flags), missingness flag generation (`has_pincode`, `has_address`, `has_name`), and open-set country cleanup (`US`, `India`, `France`).
3. **[Step 3/7] IDF Estimator Fitting**: Fit token Inverse Document Frequency (IDF) over the combined 24-million text record corpus.
4. **[Step 4/7] Candidate Generation for Test Set**: Initiated multi-key rule blocking and country-partitioned TF-IDF char 3-gram indexing across Test $S1$ entities.

---

## 2. Inventory of Completed Modules & Dataset Corrections

### A. Dataset & Directory Inconsistencies Corrected:
- **Cleaned OS Junk Files**: Removed `.DS_Store` files in `student_resource/` and `student_resource/dataset/`.
- **Synchronized Documentation Template**: Synced the complete 720-line technical methodology spec (`Documentation_template_v2.md`) to `student_resource/Documentation_template.md` (which previously held an empty template).
- **Key-Based Ground Truth Alignment**: Handled the shuffled row order of `train_ground_truth.tsv` using explicit `source1_entity_id` key indexing.
- **Imputed Missing Values**: Handled missing `business_address` (~168k in train $S2$, ~175k in train $S3$, ~129k in test $S2$, ~136k in test $S3$) with empty string fallbacks and `has_address` binary indicators.
- **Open-Set Country Support**: Designed language-agnostic features so `France` (~259k test entities) processes seamlessly without hardcoded country filters.

### B. Complete Source Code Package (`student_resource/code/business_entity_resolution/src/`):
- `data/loaders.py`: Ingests source TSVs (`sep="\t"`) and ground truth mappings safely.
- `data/normalize.py`: Vectorized text canonicalization, legal suffix extraction, address regex parsing, and missingness flags.
- `data/transliteration.py`: Mined character substitution patterns for Indian business names (`v<->w`, `s<->sh`, `i<->ee`) and custom weighted Levenshtein similarity.
- `blocking/rule_blocking.py`: Multi-key rule blocking (country + pincode, country + name prefix, country + distinctive tokens) with fast `itertuples`.
- `blocking/bm25_blocking.py`: Country-partitioned TF-IDF character 3-gram candidate retrieval.
- `blocking/recall_eval.py`: Candidate recall ceiling evaluator (>95% target) and exporter for `candidate_pairs.tsv`.
- `features/string_features.py`: Fast string metrics (Levenshtein, Token-Sort, Token-Set, Char 3-gram).
- `features/idf_jaccard.py`: IDF-weighted token overlap estimator.
- `features/structured_features.py`: PIN code exact/prefix match, street number match, suffix match/mismatch, country match, landmark flags.
- `features/interaction_features.py`: Cross-field interaction terms, composite similarity, rank and gap features.
- `models/train_ensemble.py`: GBDT Ensemble (LightGBM + CatBoost) with GPU acceleration (`device="gpu"` / `task_type="GPU"`) and scale weight class imbalance handling.
- `postprocessing/adaptive_threshold.py`: Per-entity macro $F_{0.5}$ exact grid search for decision threshold optimization.
- `postprocessing/global_assignment.py`: Greedy 1-to-many bipartite constraint solver ($S2/S3 \to \le 1\ S1$) and `matching_results.tsv` exporter.
- `pipeline.py`: Master end-to-end runner.
- `requirements.txt` & `README.md`: Reproduction instructions and environment specifications.

---

## 3. Remaining Tasks to Complete Execution

1. **Relaunch Pipeline Run**:
   - Execute `pipeline.py` to generate the full candidate set and output `output/candidate_pairs.tsv`.
2. **Feature Extraction & Model Training**:
   - Extract feature matrix for training candidate pairs.
   - Train LightGBM + CatBoost ensemble models on GPU.
3. **Threshold Tuning & Post-Processing**:
   - Find decision threshold maximizing macro $F_{0.5}$.
   - Resolve bipartite matching constraints for Test set.
   - Output final matches to `output/matching_results.tsv`.
4. **Validation Check**:
   - Execute `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test` and confirm exit code 0 (`PASS`).
5. **Submission Zip Creation**:
   - Package `output/`, `code/business_entity_resolution/`, and `Documentation_template.md` into `<team_name>_submission.zip`.
