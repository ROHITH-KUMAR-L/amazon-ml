# Business Entity Resolution — Methodology Documentation

**Challenge:** Amazon ML Challenge 2026 — Business Entity Resolution (ER)
**Team:** `<team_name>`
**Submission version:** v2.0 — extends v1.0 with iterative blocking, IDF-weighted
and cross-encoder-derived features, entity-adaptive thresholding, confidence-aware
conflict resolution, pseudo-labeling, cluster-aware validation, and a transliteration module.

---

## 1. Problem Summary

Given deduplicated Source 1 (S1) business records as the reference set, find all matching
records in Source 2 (S2) and Source 3 (S3) for every S1 entity. A given S1 entity may have
zero, one, or many true matches. Fields available: `entity_id`, `business_name`,
`business_address`, `country`. No shared identifiers exist across sources; matches must be
inferred purely from noisy name/address/country text.

Evaluation is the **macro-averaged F0.5 score**, computed per S1 entity and averaged across all
entities, which weights precision 2× over recall — false merges are penalized far more heavily
than missed matches, and correctly predicting an empty match list for a singleton is worth a
full 1.0.

Two files are required: `matching_results.tsv` (scored) and `candidate_pairs.tsv` (audited for
blocking quality, not scored). No external data, APIs, or geocoding services are permitted;
the final scoring model must be ≤8B parameters under an MIT/Apache 2.0 license.

---

## 2. Executive Summary — Performance Roadmap

Each layer below is additive on top of the previous one. This is the order components were
added and validated in, and is the recommended build order for reproduction under a time
budget: earlier rows are higher-confidence, lower-cost wins.

| Pipeline stage reached                                                         | Estimated macro-F0.5 (data-dependent) |
|----------------------------------------------------------------------------------|:---:|
| Baseline: multi-key + ANN blocking, engineered features, GBDT, fixed threshold  | ~0.72 – 0.78 |
| + IDF-weighted token Jaccard, embedding cosine feature, cross-encoder score as GBDT input | ~0.78 – 0.83 |
| + Iterative blocking refinement, entity-adaptive thresholding                   | ~0.83 – 0.88 |
| + Pseudo-labeling, cluster-aware validation split, transliteration module       | ~0.88 – 0.92 |

If time-constrained, implement top-to-bottom and stop wherever the deadline forces a cutoff —
every row above is a complete, submittable pipeline on its own.

---

## 3. High-Level Architecture

```
 S1 / S2 / S3 (raw TSV)
        │
        ▼
 ┌──────────────────────────────┐
 │ 0. Preprocessing /            │  canonicalization, address parsing, missingness
 │    Normalization              │  flags, open-set country handling,
 │    (+ transliteration module) │  data-driven transliteration similarity
 └────────┬─────────────────────┘
          ▼
 ┌──────────────────────────────┐
 │ 1. Blocking / Candidate Gen   │◄──┐  multi-key rule blocking (union) + BM25
 │    — ITERATIVE                │   │  char n-gram + domain-adapted bi-encoder ANN
 └────────┬─────────────────────┘   │  recall-ceiling measured, FN patterns mined,
          │  candidate_pairs.tsv    │  targeted rules added, re-measured (≤2 rounds)
          │                         │
          │        FN pattern mining from Stage 3 validation ──┘
          ▼
 ┌──────────────────────────────┐
 │ 2. Feature Engineering        │  string sim, IDF-weighted token Jaccard,
 │                                │  structured address, missingness flags,
 │                                │  bi-encoder cosine, cross-encoder score,
 │                                │  cross-field interactions, neighborhood/rank
 │                                │  features, transliteration similarity
 └────────┬─────────────────────┘
          ▼
 ┌──────────────────────────────┐
 │ 3. Pairwise Classification    │  diverse GBDT ensemble (name-heavy /
 │    (ensemble, imbalance-aware)│  address-heavy / full-feature) with
 │                                │  cross-encoder score as an input feature
 │                                │  (hard-negative mined), not a standalone voter
 └────────┬─────────────────────┘
          ▼
 ┌──────────────────────────────┐
 │ 4. Entity-Adaptive Threshold  │  per-entity F0.5-exact base threshold,
 │    Tuning                     │  adjusted by candidate-count / sparsity /
 │                                │  country bucket; explicit singleton rule
 └────────┬─────────────────────┘
          ▼
 ┌──────────────────────────────┐
 │ 5. Global Consistency +       │  max-weight bipartite S2/S3→≤1 S1 assignment;
 │    Confidence Propagation     │  contested "losing" entities get a singleton-
 │                                │  confidence boost rather than a blind re-match
 └────────┬─────────────────────┘
          ▼
 ┌──────────────────────────────┐
 │ 6. Pseudo-Labeling            │  very-high-confidence test predictions folded
 │    (Self-Training, ≤2 rounds) │  back into training at reduced sample weight;
 │                                │  gated on held-out validation F0.5 not regressing
 └────────┬─────────────────────┘
          ▼
 matching_results.tsv  (+ candidate_pairs.tsv from Stage 1's final round)
```

**Validation methodology note (applies throughout):** all cross-validation and threshold
tuning use a **cluster-aware entity split** (Section 9.1) rather than a naive per-S1-entity
split, to avoid partial leakage through shared S2/S3 records across folds.

---

## 4. Data Understanding & Preprocessing

### 4.1 Normalization pipeline

A single canonicalization function is applied consistently to `business_name` and
`business_address` before *any* feature is computed:

- Unicode normalization (NFKC) and lowercasing.
- Punctuation standardization (`&` → `and`, stray comma/period cleanup).
- Legal-suffix mapping to a canonical token (`corp`/`corporation` → `corp`, `pvt`/`private` →
  `pvt`, `ltd`/`limited` → `ltd`, `llc`, `inc`, etc.), with the suffix also split into a
  separate `legal_suffix` field so core-name similarity isn't penalized by suffix variation.
- Word order is **preserved** at the canonical-string level (handled instead via order-invariant
  features, Section 5.1) so both signals remain usable.
- The raw string is retained alongside the canonical one for raw-vs-raw comparison, since a
  legal-suffix mismatch is occasionally a genuine signal, not noise.

### 4.2 Address parsing

Rule-based regex parsing (no geocoding) into: street number/name, road-type normalization,
city/state/PIN segment, and a separately-extracted `landmark` field (e.g., "Near SBI ATM")
kept out of the core address string so it doesn't dilute structural similarity while still
being usable as a weak corroborating feature.

### 4.3 Missingness handling

Explicit flags — `has_pincode`, `has_state`, `has_street_number` — accompany every parsed
component so the model can distinguish "field absent" from "field present but dissimilar."

### 4.4 Open-set country handling

`country` is treated as an arbitrary string label — never one-hot encoded, never filtered.
Used as: (a) a soft blocking key with a fallback key so an unseen label never drops candidates,
(b) a binary `country_match` feature, (c) a selector for which address-abbreviation dictionary
to apply, falling back to a generic language-agnostic cleanup pass for uncovered labels (this is
what lets France flow through without errors or dropped rows).

**Edge case — country-label variants:** the open-set rule means the pipeline must not assume a
fixed vocabulary, but it should still be robust to the same country appearing under slightly
different strings within the data (e.g., `US` vs `USA` vs `United States`) if that occurs. A
light, data-driven canonicalization pass — built by clustering near-identical country strings
in the training data itself, not an external ISO country list — is applied before country is
used as a blocking key or feature, so label variance doesn't silently fragment blocks.

### 4.5 Transliteration module (new)

Indian business names in Latin script frequently have multiple valid transliterated spellings
(`Srinivas`/`Shrinivas`, `Kumar`/`Cumar`, `Sri`/`Shree`). Standard edit-distance treats these as
mismatches they aren't. Since external phonetic databases are not permitted, the substitution
patterns are **mined from the training data itself**:

- From confirmed true-match pairs (via `train_ground_truth.tsv`) where `country` is India,
  extract character-level alignment diffs between matched name pairs.
- Aggregate the most frequent substitution patterns (e.g., `v↔w`, `s↔sh`, `i↔ee`, `k↔c`) into a
  weighted substitution-cost table.
- Build a custom weighted edit-distance function using this table (lower cost for known
  transliteration substitutions, standard cost otherwise) and expose it as an additional
  similarity feature, applied wherever `country == "India"` for either record in the pair (and
  available, at lower expected value, elsewhere as a general fuzzy fallback).
- This table is training-data-derived only — no external transliteration dictionaries or APIs —
  keeping it compliant with the no-external-data rule.

---

## 5. Stage 1 — Candidate Generation (Blocking), Iterative

Blocking sets the **recall ceiling** for the whole pipeline. This version treats blocking as an
iterative process, not a single fixed pass.

### 5.1 Round 1 — Multi-key rule blocking (union) + hybrid retrieval

- **Rule blocking (union of independent passes, not an AND condition)**: same-country + same
  PIN (when present), same-country + matching city/state token, Metaphone of the *sorted* name
  token set (order-invariant), character 3–4-gram MinHash/LSH bucket on the canonical name.
- **BM25 over character n-grams** on concatenated normalized name+address — naturally captures
  digit substrings (PIN/street numbers) as well as text tokens in one index.
- **Dense ANN retrieval** using a multilingual bi-encoder (Section 5.2) over the same
  concatenated text via FAISS, top-K per S1 entity.
- All four sources are **unioned**, not intersected, favoring recall at this stage.

### 5.2 Domain-adapted bi-encoder (new)

A general-purpose multilingual embedding model is a reasonable starting point, but business
entity strings are a narrow subdomain (legal suffixes, landmark phrases, PIN-heavy addresses)
that a generically-trained encoder doesn't specialize in. The bi-encoder used for ANN retrieval
is **contrastively fine-tuned on the training pairs only**:

- Positive pairs: confirmed matches from `train_ground_truth.tsv` (name+address text of both
  records).
- Negative pairs: random same-country non-matches, plus hard negatives harvested from Round 1's
  own false-positive candidates (high retrieval score, confirmed non-match in ground truth).
- A small number of fine-tuning epochs (low learning rate) to adapt the embedding space to this
  domain's noise patterns without catastrophically forgetting the model's general multilingual
  competence — this matters specifically for France, where the model must still generalize.

### 5.3 Recall-ceiling measurement (gate before Stage 2)

On the cluster-aware validation split (Section 9.1):

- **Blocking recall** = fraction of true S2/S3 matches present anywhere in the S1 entity's
  candidate set.
- **Reduction ratio** = 1 − (candidate pairs / all possible S1×(S2∪S3) pairs).

If blocking recall is below target, proceed to Section 5.4 rather than moving on with a capped
recall ceiling — no amount of downstream modeling can recover a true match blocking never
retained.

### 5.4 Round 2 — Targeted refinement from false-negative mining (new)

After a first full pass through Stage 3 (classification) on validation data, the false
negatives are inspected specifically for **candidate-set absence** vs. **classifier miss**:

- If a true match never appeared in `candidate_pairs.tsv` for its S1 entity, it's a blocking
  failure, not a classifier failure. These cases are clustered by noise pattern (e.g., PIN
  format mismatch, extreme abbreviation, transliteration variant, landmark-only address).
- A targeted blocking rule is added per recurring pattern (e.g., a first-N-character + city
  block for cases where full-name phonetic blocking fails due to heavy abbreviation).
- Recall ceiling and reduction ratio are re-measured. This loop is run **at most twice** — after
  that, diminishing returns and overfitting the blocking rules to validation-specific noise
  outweigh further iteration, and remaining recall loss is treated as an accepted limitation
  (Section 12).

### 5.5 Output

Final-round surviving candidates, per S1 entity, are written to `candidate_pairs.tsv` — the
exact set Stage 3 scores. Every ID in `matching_results.tsv` is guaranteed to appear here.

---

## 6. Stage 2 — Feature Engineering

### 6.1 String similarity features (name & address, computed separately)

Levenshtein (normalized), Jaro-Winkler, token-sort/token-set ratio (order-invariant), TF-IDF
cosine similarity at both word level and character n-gram level.

### 6.2 IDF-weighted token Jaccard (new, high priority)

Plain Jaccard treats every token equally, so common tokens (`pvt`, `store`, `trading`) inflate
similarity for genuinely unrelated businesses. Token overlap is instead weighted by each
token's inverse document frequency computed over the full training+test name corpus (fit only
on non-ground-truth-dependent statistics, so this is safe to compute over test text too):
rare, distinctive tokens (a specific brand or family name) contribute far more to the similarity
score than generic business vocabulary. This is applied to both name and address tokens.

### 6.3 Token-level overlap

Standard Jaccard (name and address), with common legal suffixes stripped beforehand — retained
alongside the IDF-weighted version since the two carry different information (raw overlap
volume vs. discriminative overlap).

### 6.4 Structured address features

Street-number exact/fuzzy match, PIN exact/prefix match, state/city exact match, landmark
phrase overlap (low weight).

### 6.5 Categorical & missingness features

`country_match`, `has_pincode`/`has_state`/`has_street_number` for both records plus their
conjunction (both present / one missing / both missing).

### 6.6 Embedding-derived features (new)

- **Bi-encoder cosine similarity**: the actual cosine similarity between the domain-adapted
  bi-encoder's embeddings of the two records — a calibrated similarity, distinct from the
  retrieval-rank signal used for blocking.
- **Cross-encoder score as a GBDT input feature** (not a standalone classifier — see Section
  7.2): the fine-tuned cross-encoder's match probability for the pair is added as one more
  column in the feature vector, letting the GBDT learn how much to trust it relative to the
  symbolic features rather than treating it as an independent vote.

### 6.7 Cross-field interaction features (new)

Explicit products/combinations the GBDT would otherwise have to approximate through many split
levels: `name_sim × address_sim`, `name_sim × country_match`, `high_name_sim AND pin_mismatch`
(flag for the "same name, wrong location" pattern), `name_sim × has_pincode` (down-weight
address disagreement when PIN data is simply absent rather than actually different).

### 6.8 Neighborhood / rank features (new)

Computed per S1 entity across all its candidates, then attached back to each pair as context the
per-pair view otherwise lacks: `rank_among_candidates` (this pair's score rank), `score_gap_to_
second_best`, `max_candidate_score_for_entity`, `candidate_count_for_entity`. A pair scoring
0.85 when it's the clear best of two candidates is a different situation from scoring 0.85 amid
five near-tied candidates, and the classifier/threshold stage should be able to see that
difference.

### 6.9 Transliteration similarity feature

Output of the custom weighted edit-distance function from Section 4.5, applied per pair.

### 6.10 Legal-suffix / raw-string agreement

Low-weight feature indicating whether the *raw*, pre-stripped legal designation matches.

---

## 7. Stage 3 — Pairwise Classification

### 7.1 Ensemble architecture (revised for diversity, not duplication)

Rather than training two GBDT variants (e.g., LightGBM and CatBoost) on an identical feature
set — which tend to agree closely and add limited value — the ensemble is built for genuine
**evidence diversity**:

- **Model A (full-feature)**: trained on the complete feature vector (Sections 6.1–6.9).
- **Model B (name-heavy)**: trained primarily on name-similarity and token features, address
  features down-weighted/excluded — captures cases where address data is sparse or unreliable.
- **Model C (address-heavy)**: the inverse — captures cases where name variation is extreme
  (heavy abbreviation, DBA) but address is a strong, complete signal.
- Final probability is a weighted blend of the three (weights tuned on the cluster-aware
  validation split), which is more robust to any single feature group being noisy or missing
  for a given pair than one model trained on everything at once.

### 7.2 Cross-encoder: hard-negative-mined, used as a feature not a voter

- **Bootstrap pass**: Model A is first trained *without* the cross-encoder feature, to get an
  initial probability ranking.
- **Hard negative mining**: candidate pairs the bootstrap model scores highly (near or above the
  eventual threshold band) that are confirmed non-matches in ground truth become the
  cross-encoder's negative training examples, alongside the standard true-positive matches. This
  specifically teaches the cross-encoder to discriminate the *hard* cases rather than trivial
  random negatives, which is where a fine-tuned transformer earns its keep on a small positive
  set.
- The fine-tuned cross-encoder then scores every candidate pair, and that score is folded back
  in as an input feature (Section 6.6) for the final ensemble retraining — not run as a separate
  full classifier competing with the GBDTs.

### 7.3 Class imbalance handling

`scale_pos_weight` (or equivalent) set from the true positive/negative ratio in the training
candidate set, applied to each ensemble member independently.

### 7.4 Fitting discipline

All data-dependent transforms — TF-IDF vectorizers, IDF weighting statistics, ANN indices, the
bi-encoder fine-tuning, the transliteration substitution table — are fit **only on the training
fold** during validation. Fitting on data that includes the validation ground truth inflates the
offline F0.5 estimate and produces a threshold/ensemble blend that won't generalize.

---

## 8. Stage 4 — Entity-Adaptive Threshold Tuning

### 8.1 Baseline: exact per-entity F0.5 replication

Because F0.5 here is a macro-average computed **per S1 entity, then averaged**, threshold search
replicates the exact scoring procedure rather than approximating it via pooled precision/recall:
for a candidate threshold, matches are derived, F0.5 is computed per entity, and per-entity
scores are averaged — the same computation the leaderboard performs.

### 8.2 Entity-adaptive adjustment (new)

A single global threshold is a reasonable floor but leaves accuracy on the table, because the
optimal cutoff systematically differs by entity context:

- Entities with **many candidates** (common, generic names) warrant a stricter threshold — more
  competition for the match means higher confidence is needed to justify a positive.
- Entities in **sparse-data regions** (missing PIN/state, landmark-only addresses — common in
  parts of the Indian data) systematically produce lower similarity scores even for true
  matches, warranting a slightly relaxed threshold conditioned on the missingness flags.
- Entities with a **large score gap** between the best and second-best candidate (Section 6.8)
  can tolerate a lower absolute threshold safely, since the decision is unambiguous relative to
  alternatives.

Implementation: entities are bucketed by `candidate_count`, `country`, and a sparsity indicator
derived from the missingness flags; a per-bucket threshold is tuned via the exact per-entity
F0.5 procedure (Section 8.1) on the cluster-aware validation split. Buckets with too few
validation examples to tune reliably fall back to the global threshold, to avoid overfitting a
threshold to a handful of validation entities.

### 8.3 Singleton rule

If the maximum predicted probability across all candidates for an entity falls below its
(possibly bucket-adjusted) threshold, an empty match list is predicted — capturing the full 1.0
credit correctly-identified singletons receive.

### 8.4 Domain-shift check (France)

Threshold and bucket robustness is stress-tested by holding out one full training country during
a validation pass and treating it as unseen, as the closest available proxy for how the France
subset of the test set will behave, since no French training labels exist.

---

## 9. Stage 5 — Global Consistency + Confidence Propagation

### 9.1 Motivation

Since S1 is deduplicated, a given S2/S3 record should realistically match at most one S1 entity.
Independent per-pair thresholding has no mechanism preventing two different S1 entities from
both clearing threshold against the same S2 record — and since F0.5 penalizes false merges twice
as heavily as misses, resolving this is high-leverage.

### 9.2 Bipartite assignment

A weighted bipartite graph (S1 entities vs. S2/S3 records, edge weight = calibrated match
probability, restricted to edges that cleared Stage 4's threshold) is resolved via greedy
max-weight assignment, with an exact Hungarian-style solve as a fallback for smaller, densely
contested subgraphs. The constraint is enforced only in the S2/S3→S1 direction (each S2/S3
record maps to at most one S1 entity); an S1 entity may retain multiple surviving matches, since
one true entity can legitimately have several real matches across S2/S3.

### 9.3 Confidence propagation for contested losers (new)

When an S1 entity loses a contested edge (its best candidate was awarded to a competing S1
entity with a higher score), the pipeline does not simply fall through to that entity's
next-best remaining candidate by default. Instead, losing a contested edge is treated as
**evidence toward singleton status**: if the entity's next-best remaining score is itself
unremarkable, it's a signal the original high score was driven by resemblance to a specific
other business rather than genuine ambiguity among several plausible matches for this entity —
and the entity is more conservatively evaluated against its (adjusted, e.g. slightly raised)
threshold before accepting the fallback candidate. This prevents false merges from cascading
through the conflict-resolution step.

### 9.4 Franchise / multi-branch edge case (see also Section 11.3)

Bipartite S2/S3→≤1-S1 enforcement is correct for genuine duplicates, but must not be misapplied
to legitimately distinct branches of the same chain (e.g., two different "Starbucks" S1 entities
at different addresses, each with a real, distinct S2/S3 counterpart). Address-based
disambiguation features (Section 6.4, 6.7) are what keep these separated *before* the bipartite
step ever sees them as competing for the same record — the conflict-resolution step only fires
when two S1 entities are actually contesting the *same* S2/S3 record, which correctly-blocked
distinct branches should not do in the first place.

---

## 10. Stage 6 — Pseudo-Labeling / Self-Training (new)

### 10.1 Procedure

1. Run the full pipeline (Stages 1–5) once to produce a complete first-pass `matching_results`
   on the test set.
2. Select **very-high-confidence** predictions only (e.g., calibrated probability above a strict
   cutoff, well above the operating threshold — conservatively chosen, not the same as the
   decision threshold) as pseudo-positive pairs, and high-confidence resolved singletons
   (max candidate score well below threshold) as implicit pseudo-negatives for their contested
   candidates.
3. Add these pseudo-labeled pairs into the training set for a second round of feature-statistic
   fitting and ensemble retraining, at a **reduced sample weight** (e.g., half the weight of a
   real ground-truth label) so the model treats them as corroborating rather than equally
   trustworthy evidence.
4. Re-run Stages 1–5 with the retrained ensemble.

### 10.2 Safety gate

Before accepting a pseudo-labeling round's output for submission, held-out cluster-aware
validation F0.5 (computed on real ground truth only, never pseudo-labels) must be **equal to or
better** than the pre-pseudo-labeling round. If it regresses, the round is discarded and the
prior model is kept. This is run for **at most two rounds** — confirmation bias and drift risk
compound with further iteration, and gains beyond two rounds are typically marginal.

---

## 11. Edge Cases and How They're Handled

This section catalogs failure modes considered during design, grouped by where in the pipeline
they'd otherwise cause silent errors, dropped entities, or false merges.

### 11.1 Data quality edge cases

- **Null / empty `business_name` or `business_address`**: normalization treats these as a
  missing field (not an empty string to compare), setting the relevant missingness flag rather
  than letting a blank-vs-blank comparison register as a false high similarity.
- **Malformed TSV rows** (embedded tabs, unescaped characters): loader validates column count
  per row on ingest and quarantines/logs malformed rows rather than silently misaligning
  columns; since commas are expected inside address/ID-list fields, the parser never splits on
  commas.
- **Unicode/encoding anomalies** (mixed scripts, stray control characters, emoji): NFKC
  normalization plus a defensive strip of non-printable characters before any similarity
  computation, so a single malformed byte doesn't crash a batch.
- **Field-swapped records** (address text accidentally in the name column or vice versa):
  a lightweight heuristic check (e.g., presence of digit-heavy tokens or address keywords in the
  name field) flags likely swaps for manual review rather than silently feeding corrupted
  features to the model.

### 11.2 Blocking edge cases

- **Extremely generic/common names** ("Corner Store", "ABC Traders"): these inflate candidate
  counts sharply. A per-entity candidate cap (top-K by combined BM25+ANN score) bounds runtime
  and reduces the chance the classifier is asked to discriminate among dozens of near-identical
  generic-name candidates without enough signal to do so reliably — this interacts directly with
  the entity-adaptive threshold (Section 8.2), which applies a stricter cutoff exactly for these
  high-candidate-count entities.
- **Blocking explosion / runtime**: reduction ratio (Section 5.3) is monitored specifically to
  catch cases where a blocking key is too permissive for the dataset scale; the cap above is the
  practical backstop.
- **Zero candidates after blocking**: handled identically to a low-confidence classifier
  outcome — an explicit empty match list, not a pipeline error, and every S1 test entity still
  gets a row in both output files.

### 11.3 Matching-logic edge cases

- **Franchise/chain businesses with multiple legitimate branches**: covered in Section 9.4 —
  distinct branches must be kept separated by address-based features *before* the global
  consistency step, since it only resolves genuine contests for the same record.
- **Same address, different business** (e.g., a shared building or mall directory entry): a
  cross-field interaction feature specifically flags "high address similarity, low name
  similarity" so the classifier doesn't over-weight shared location as a proxy for identity.
- **Numeric-heavy / branch-numbered names** ("Store #4521"): the structural digit-match feature
  (Section 6.4/6.7) is what disambiguates these rather than relying on string similarity alone,
  since a one-digit difference in a branch number can otherwise look like a near-perfect text
  match.
- **Very short / low-information names** (single generic word): flagged via a name-length /
  token-count feature so the model can down-weight name evidence and lean on address and
  structural features instead.

### 11.4 Post-processing edge cases

- **Cascading false merges through conflict resolution**: addressed by confidence propagation
  (Section 9.3) rather than a naive fallback-to-next-candidate rule.
- **Ties in bipartite assignment weights**: broken deterministically (e.g., by a stable
  secondary key such as entity_id) so output is reproducible across runs, not by arbitrary
  dict/set ordering.

### 11.5 Compliance / format edge cases

- **Every S1 test entity must appear exactly once** in both output files, including zero-match
  entities with an empty field — enforced as a final pipeline assertion, not left to downstream
  validation to catch.
- **No duplicate entity IDs within an ID list, no cross-set IDs** (a matched ID must exist in
  the test set and be S2-/S3-prefixed) — enforced programmatically before writing output, in
  addition to running `utils/validate_submission.py` as a final gate.
- **`candidate_pairs.tsv` must be a superset of `matching_results.tsv`** for every entity, by
  construction, since matches are only ever selected from that entity's final-round candidates.

### 11.6 Computational / reproducibility edge cases

- **ANN approximate search non-determinism**: FAISS index construction and search use fixed
  random seeds where the library allows it; result stability is checked across repeated runs on
  a fixed sample before trusting the recall-ceiling measurements.
- **GBDT and fine-tuning non-determinism**: all training entry points accept and log a fixed
  random seed; reported validation metrics are averaged across at least two seeds to avoid
  reporting a favorable-but-unstable single run.
- **Runtime budget**: the bi-encoder and cross-encoder fine-tuning steps, and full-corpus
  cross-encoder scoring, are the dominant cost centers; both are logged with wall-clock time on
  representative data volume so the two-round blocking iteration and two-round pseudo-labeling
  loop can be scoped down (fewer rounds) if the overall time budget is tight, per the roadmap in
  Section 2.

---

## 12. Validation Strategy & Metrics

### 12.1 Cluster-aware entity split (revised)

A naive per-S1-entity split can leak information: if S1-001 and S1-002 both legitimately or
spuriously relate to the same S2/S3 record in the training graph, splitting one to train and the
other to validation lets validation partially "see" that record's characteristics during
training. Instead:

- Build a graph over all training entity IDs (S1, S2, S3) with edges from `train_ground_truth`.
- Identify connected components.
- Assign entire components — never partial components — to either the train or validation fold,
  stratified to preserve the overall singleton/multi-match ratio across folds.

### 12.2 Metrics tracked at each stage

- Stage 1: blocking recall (ceiling), reduction ratio, per-round delta across iterations.
- Stage 3: pairwise precision/recall/AUC per ensemble member (diagnostic).
- Stage 4: macro-averaged F0.5 via the exact per-entity procedure, both globally and per
  threshold bucket.
- Stage 5: false-merge count before/after conflict resolution and confidence propagation.
- Stage 6: validation F0.5 delta per pseudo-labeling round (gate for acceptance).

### 12.3 Local validation

`utils/validate_submission.py` is run against every candidate submission before it's considered
for leaderboard upload.

---

## 13. Compliance & Fair Play

- No external APIs, commercial ER services, government registries, or geocoding services are
  used at any stage; the transliteration substitution table and country-label canonicalization
  are both mined exclusively from the provided training/test data, not an external dictionary.
- The final scoring model(s) are ≤8B parameters and MIT/Apache-2.0 licensed.
- All preprocessing statistics (TF-IDF vocabularies, IDF weights, ANN indices, phonetic and
  transliteration tables) are fit exclusively from the provided training and test source files.

---

## 14. Project Structure

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── data/
│       │   │   ├── loaders.py              # TSV loading, malformed-row quarantine
│       │   │   ├── normalize.py            # canonicalization, address parsing,
│       │   │   │                           #   missingness flags, country handling
│       │   │   └── transliteration.py      # NEW: mined substitution table + weighted
│       │   │                               #   edit distance (Section 4.5)
│       │   ├── blocking/
│       │   │   ├── rule_blocking.py        # multi-key union blocking
│       │   │   ├── bm25_blocking.py        # char n-gram BM25 index
│       │   │   ├── biencoder_blocking.py   # NEW: contrastive fine-tuning + ANN (5.2)
│       │   │   ├── recall_eval.py          # recall-ceiling / reduction-ratio checks
│       │   │   └── fn_mining.py            # NEW: false-negative pattern mining (5.4)
│       │   ├── features/
│       │   │   ├── string_features.py      # 6.1
│       │   │   ├── idf_jaccard.py          # NEW: 6.2
│       │   │   ├── structured_features.py  # 6.4
│       │   │   ├── embedding_features.py   # NEW: bi-/cross-encoder features (6.6)
│       │   │   ├── interaction_features.py # NEW: 6.7
│       │   │   └── neighborhood_features.py# NEW: rank/gap features (6.8)
│       │   ├── models/
│       │   │   ├── train_ensemble.py       # NEW: Model A/B/C diverse ensemble (7.1)
│       │   │   ├── cross_encoder.py        # NEW: hard-negative-mined reranker (7.2)
│       │   │   └── calibration.py
│       │   ├── postprocessing/
│       │   │   ├── threshold_search.py     # per-entity F0.5-exact sweep
│       │   │   ├── adaptive_threshold.py   # NEW: entity-bucket adjustment (8.2)
│       │   │   └── global_assignment.py    # bipartite + confidence propagation (9.3)
│       │   ├── selftraining/
│       │   │   └── pseudo_label.py         # NEW: Stage 6 loop + validation gate (10.2)
│       │   ├── validation/
│       │   │   └── cluster_split.py        # NEW: connected-component split (12.1)
│       │   ├── evaluate.py                 # replicates leaderboard macro-F0.5 locally
│       │   └── pipeline.py                 # end-to-end orchestration
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

---

## 15. Reproducibility Instructions (summary)

```bash
# 1. Environment
pip install -r requirements.txt

# 2. Preprocess (+ mine transliteration table from training positives)
python -m src.data.normalize --in dataset/train --in dataset/test --out data/normalized
python -m src.data.transliteration --ground-truth dataset/train/train_ground_truth.tsv \
  --out data/translit_table.json

# 3. Blocking round 1
python -m src.blocking.rule_blocking --data data/normalized --out output/candidate_pairs.tsv
python -m src.blocking.bm25_blocking --data data/normalized --out output/candidate_pairs.tsv --union
python -m src.blocking.biencoder_blocking --data data/normalized --ground-truth dataset/train/train_ground_truth.tsv \
  --out output/candidate_pairs.tsv --union
python -m src.blocking.recall_eval --candidates output/candidate_pairs.tsv --val-fold data/val_fold.parquet

# 3b. Blocking round 2 (targeted, up to once more) — only if recall ceiling below target
python -m src.blocking.fn_mining --candidates output/candidate_pairs.tsv --val-fold data/val_fold.parquet \
  --out data/blocking_gap_report.json
# ... add targeted rule per report, re-run 3, re-check recall_eval

# 4. Feature engineering
python -m src.features.string_features --candidates output/candidate_pairs.tsv --out data/features.parquet
python -m src.features.idf_jaccard --candidates output/candidate_pairs.tsv --out data/features.parquet --append
python -m src.features.structured_features --candidates output/candidate_pairs.tsv --out data/features.parquet --append
python -m src.features.embedding_features --candidates output/candidate_pairs.tsv --out data/features.parquet --append
python -m src.features.interaction_features --features data/features.parquet --append
python -m src.features.neighborhood_features --features data/features.parquet --append

# 5. Cross-encoder: bootstrap → hard-negative mine → fine-tune → re-score → append feature
python -m src.models.train_ensemble --features data/features.parquet --bootstrap-only
python -m src.models.cross_encoder --features data/features.parquet --mine-hard-negatives
python -m src.features.embedding_features --candidates output/candidate_pairs.tsv --out data/features.parquet \
  --append --cross-encoder-scores

# 6. Final ensemble
python -m src.models.train_ensemble --features data/features.parquet

# 7. Threshold search (per-entity F0.5-exact + adaptive buckets) + global consistency
python -m src.postprocessing.threshold_search --val-fold data/val_fold.parquet
python -m src.postprocessing.adaptive_threshold --val-fold data/val_fold.parquet
python -m src.postprocessing.global_assignment --scored data/scored_candidates.parquet --out output/matching_results.tsv

# 8. Pseudo-labeling (≤2 rounds, each gated on validation F0.5 not regressing)
python -m src.selftraining.pseudo_label --round 1 --confidence-cutoff 0.97
# re-run steps 6-7 with augmented training set; compare validation F0.5 before accepting

# 9. Validate before submitting
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

---

## 16. Known Limitations & Future Work

- **Greedy global assignment** remains an approximation for large contested subgraphs; exact
  solving is reserved for smaller components for tractability.
- **Country-specific abbreviation dictionaries** exist only for US/India; France and any further
  unseen country fall back to generic normalization — safe, but a weaker signal than a dedicated
  dictionary.
- **Iterative blocking is capped at two rounds** by design (Section 5.4) to avoid overfitting
  blocking rules to validation-specific noise; residual recall loss beyond that point is an
  accepted, documented limitation rather than chased indefinitely.
- **Pseudo-labeling is capped at two rounds** (Section 10.2) for the same reason — confirmation
  bias risk compounds with further self-training iterations.
- **The transliteration substitution table** is only as good as the volume and diversity of
  India-labeled true-match pairs in the training set; a small training set limits how many
  substitution patterns can be reliably mined.
- **Entity-adaptive threshold buckets** fall back to the global threshold when validation
  examples in a bucket are too sparse to tune reliably — this is a deliberate conservatism
  trade-off, accepting a slightly less optimal threshold over the risk of overfitting to a
  handful of bucket-specific validation entities.
