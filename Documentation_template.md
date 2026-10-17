# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** git push --force  
**Team Members:** Abhinav (solo)  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary
Two stages. First, per-country TF-IDF retrieval gives each S2/S3 record a short candidate list. It is the union of record→S1 top-10, S1→record top-10 and address-only top-3. **Adaptive pruning** (a rank + cosine-gap rule) keeps the set small: 10.54 candidates per S1 on test (18.25M pairs over 1.73M S1). Indic-script names are first mapped to Latin with a token dictionary learned from train pairs. Then a LightGBM pairwise matcher with 60 features scores the list, and each record goes to its **single** max-probability S1 if prob ≥ t. The main findings: 95% of the baseline's false positives were **synthetic distractors** (near-duplicates of a real S1 with a shifted house number and an extra name token), and the main recall loss was Indic-script names that shared no n-grams with their Latin S1. Grouped-holdout macro F0.5 rose from **0.753 → 0.969 → 0.974**; the final submission (sub-04b) scores **0.965648** on the public LB.

---

## 2. Methodology

### 2.1 Problem Analysis
Full stats are in `docs/EDA.md`; errors are in `docs/ERRORS.md`.
- **One S1 per record:** 0 of 7.64M train GT pairs reuse an S2/S3 ID across S1 entities, so the task is really an *assignment* problem (record → ≤1 S1). About **26% of S2/S3 records are never matched** (distractors). 94% of S1 have matches (mean 3.46, S2/S3 split 48/52), so the all-empty baseline scores only 0.056.
- **Country never differs** within a matched pair (0/7.64M), so everything is partitioned by the `country` string as an open set. Nothing is hard-coded.
- **Postal codes are nearly absent:** ≤1.2% of India/France addresses and 11% of US addresses have one, and only 8% of US pairs share one. They can't be a blocking key.
- **Script mismatch (India):** S1 is 100% Latin, but 18% of India train S2/S3 names (753k records; S2 24%, S3 13%) are in an Indic script: mostly Devanagari, plus Telugu, Kannada, Malayalam and Bengali. 22% of India true pairs have name similarity < 50. The Indic names use a small vocabulary (1,312 token types in all train pairs), which makes a learned token dictionary effective (§3).
- **Hard positives:** DBA/brand names that differ completely at the same address, empty S2/S3 addresses (~3%), zero-padded or extra numbers (`0971`↔`971`, PMB/unit).
- **France is unseen (test only, 15% of test S1):** legal suffixes (SARL/SAS/EURL/SASU/SCI, `S.A.R.L.`), injected diacritic noise (`Grânde`, `Çlub`), département vs region (`Gironde` vs `Nouvelle-Aquitaine`), and `R.`/`AV`/`N°`/`BIS` abbreviations. So the pipeline uses no country feature, language-agnostic normalization, and one global threshold.
- **Test shift:** 5.75 S2+S3 records per S1 in test vs 4.68 in train, which likely means more distractors. The threshold therefore has to hold up under missing S1s (§4).

### 2.2 Solution Strategy
**Approach Type:** Blocking (multi-view TF-IDF retrieval) + gradient-boosted pairwise classifier + constrained one-to-one assignment  
**Core Innovation:** (1) Distractor-aware features: address-number drift, legal-suffix/extra-token deltas, sibling/graph context. (2) A per-record argmax that uses the one-S1-per-record structure. (3) An Indic→Latin token dictionary learned only from train-split pairs, so Indic records get meaningful name vectors and name features. (4) S1-side retrieval, which recovers records crowded out of their own top-k. The threshold was chosen with a distractor stress test instead of only in-distribution validation.

**Validation protocol:** holdout grouped by S1 entity (hash-based, 20%). A val S1 sample is drawn per country (seed 42). *fit* = ~1.5M records with no val S1 among their candidates and whose true S1 is not in val. *val* = every record that has a val-S1 candidate or whose true S1 is in val (exact per-S1 attribution). Early stopping uses a record-grouped 10% of *fit*, never val. The Indic dictionary is learned from non-val S1 pairs only. Nothing is tuned on test.

---

## 3. Candidate Generation (Blocking)
- **Indic name mapping (D-014):** an Indic-token → Latin-token dictionary learned from train-split GT pairs (Indic record name vs Latin S1 name with equal token count, aligned in order; count ≥ 3, share ≥ 0.7). It has 1,281 entries and covers 95.5% of Indic token occurrences in val and 91.7% in test. Unmapped tokens are transliterated. The mapped name replaces the record's name core everywhere downstream. On sampled val pairs whose Indic record was missed by blocking, dictionary name + address retrieval recovers **94%**, vs 2% for plain transliteration and 9% for a consonant skeleton.
- **Keys:** per-country TF-IDF fit on S1 only. Char-3-grams of the *name core* (legal suffixes removed) plus word tokens of the address, max_df 5%. Each view is L2-normalized, the views are hstacked, and the result is normalized again.
- **Three retrievals, unioned (D-015):** record → S1 top-10 (combined); S1 → record top-10 (combined), which recovers records whose own top-k is crowded by look-alike S1s; and record → S1 address-only top-3. All three are two-stage (D-008): top-N by rare features only (df ≤ 5000), then rescored with the full vectors, at about 1 ms per S1 query. From sub-04b on, index order is hash-permuted so exact-cosine ties break pseudo-randomly, not by ID order (§4, audit).
- **Adaptive pruning (the mechanism that keeps the candidate set small):** pruning with **delta = 0.1** and **K = 10** is what keeps the test candidate set at **10.54 per S1** instead of the full three-way union. Keep a pair if (record rank < K and (rank 0 or top-1 cos − cos ≤ delta)) or S1-side rank < 5. The rule was swept on train over delta, K and the per-source ranks. Records whose top-1 S1 clearly outscores the rest keep only that record-side candidate, while ambiguous records keep every candidate within the delta gap (up to K). S1-side top-5 hits are always kept. On the val sample the same rule gives 8.6 per S1.
- **Recall (train, 200k val-sample S1):** pair recall **0.9701** (India 0.9574, US 0.9828) at **8.6 candidates per S1 (val sample)** for the final sub-04b lists (sub-04 with ID-ordered ties: 0.9653 at 8.5). sub-03's pruned top-5 lists gave 0.9524 (India 0.9349, US 0.9699) at 19.4.
- **Candidate pairs generated (test):** 18,251,530 across 1,732,544 S1 entities, i.e. **10.54 candidates per S1** on average (median 8). 243 S1 have no candidates. These are exactly the pruned lists the matcher scored, written to `candidate_pairs.tsv`.
- **Not losing matches:** recall was measured per country and per slice (Indic, empty address, DBA). The remaining blocking misses are mostly empty-address records (slice recall 0.77): generic names shared by many S1 can't be resolved without an address. An extra exact name-only search for them (sub-06, D-019) moved slice recall only to 0.772 and did not improve the LB.

---

## 4. Matching Model

**Features (60, `features.py`; no country, ID, row-position or GT columns):**
| group | # | features |
|---|---|---|
| Retrieval / rank | 12 | cos, cos_top, rank, n_cand, gap_top, gap_next, gap_12 (top1–top2), tie, tie_12, **retrieval-source ranks** (record-side, address-only, S1-side) |
| Name vs address views | 6 | name-only cos, address-only cos, diff, product, name-hi/addr-lo, addr-hi/name-lo |
| Name | 14 | token_set, token_sort, partial, Jaro-Winkler, full-name ratio, **translit skeleton ratio** (hand-written Indic→Latin + consonant skeleton, `translit.py`), exact name, exact core, extra tokens rec / S1, legal-suffix equal / rec / S1, Indic flag |
| Address | 13 | token_set, non-numeric token_set, numbers without leading zeros: first equal, all equal, S1 ⊆ rec, shared, rec-only, S1-only, #rec, #S1, **log min \|diff\| of unmatched numbers**, log first diff, rec address empty |
| Graph / siblings | 8 | S1 in-degree (all / S2 / S3) from *predicted* rank-0 pointers, other-source sibling, cos rank among pointers, cos vs S1's best pointer, max name / address token_set to siblings (the sibling-linking signal) |
| Lengths / source | 7 | name-core / address length and token counts (rec, S1), source |

Top gain (sub-04b): **num_min_logdiff 31%** (house-number drift is the distractor signature), rank 11%, num_s1_only 7.5%, record-side retrieval rank 7.2%, sib_max_addr 6.9%, num_s1_subset 5.7%, full-name ratio 4.2%. The split among the correlated address-number features varies between retrains.

**Model type:** LightGBM binary (lr 0.05, 127 leaves, min_data_in_leaf 200). Trained on 1.82M pairs / 1.5M records (pos rate 0.619), with early stopping at iteration 1949 (holdout logloss 0.0265; sub-04b).  
**Assignment:** for each record, take the **max-probability candidate** and assign it if prob ≥ t. This enforces one S1 per record by construction.  
**Threshold selection:** maximize macro F0.5 on val, *and* check it against a **distractor stress test**. The stress test removes 20% of the val S1 sample (seed 43) from all candidate lists, which turns their true records into distractors that have plausible wrong neighbours. Rank, gap, graph and sibling features are recomputed for it. Results:
- Val macro F0.5 stays flat (≥ 0.9739 for t ∈ [0.66, 0.77]). **t = 0.73** (sub-04b) gives val 0.9740 and stress 0.9741 (P 0.994 / R 0.943; sub-04 at t = 0.71: 0.9737 / 0.9737), so the choice isn't sensitive to extra distractors or missing S1s. One global t is used because France has no labels.
- The primary output uses t = 0.73 with no output size cap (D-012).
- **Group-level joint assignment (D-018)** was tried on top: accept or reject whole predicted clusters by size-dependent mean-probability thresholds, and re-route rejected members to their second-best S1. It was tuned on half the val S1 and gave no gain on the held-out half (0.97429 vs 0.97435). The graph/sibling features already carry the group context.

**Leakage audit:**
- `matcher.py features` asserts that the model contains no ID / position / GT / country columns, and reports each feature's |Spearman| against record ID and S1 ID on 1M val pairs. For sub-03: max 0.0035 with rec ID, 0.0058 with S1 ID; label vs IDs ≤ 0.0009.
- For sub-04 the audit flagged 0.168 (n_cand vs rec ID) and label vs rec ID 0.040. Cause: exact cosine ties are common (31% of pairs), and top-k kept tied items by index position, with indexes in ID order. It is not GT leakage (IDs are never features, and val and test follow the same process), but it made the candidate set depend on ID order. The fix (sub-04b) hash-permutes the retrieval indexes and breaks ties by a pair hash. Its audit is clean again: max |Spearman| 0.0041 with rec ID (indic), 0.0070 with S1 ID (n_partial); label vs rec ID 0.0011, vs S1 ID 0.0001. (log: work/b04b_audit_console.log).
- Data-order check on train (`diagnose.py`): corr(row position rec, S1) ≈ +0.003, corr(ID rec, ID S1) ≈ 0, match rate flat across file-position and ID deciles (S2 0.732–0.735, S3 0.746–0.747), P(next row has same S1) = 0. **The data has no order/ID signal, and none is used.**
- Graph/sibling features use *predicted* rank-0 pointers only, never GT. Fit and val records are disjoint by S1 group. The Indic dictionary never sees val pairs.

---

## 5. Results & Error Analysis

| version | method | val P | val R | **val macro F0.5** | public LB |
|---|---|---|---|---|---|
| sub-01 | all empty | – | 0 | 0.056 (train, analytic) | – |
| sub-02 | top-1 S1 + cos ≥ 0.675 | 0.809 | 0.800 | **0.753** (India 0.716, US 0.777) | – |
| sub-03 | LightGBM + per-record argmax, t = 0.69 | 0.993 | 0.933 | **0.969** (India 0.958, US 0.979); stress 0.969 | 0.960403 |
| sub-04 | + Indic dictionary, blocking v2, 60 features, t = 0.71 | 0.994 | 0.943 | **0.974** (India 0.967, US 0.981); stress 0.974 | **0.965297** |
| **sub-04b (final)** | sub-04 with ID-independent tie-breaking (D-017), t = 0.73 | 0.994 | 0.943 | **0.9740** (India 0.9672, US 0.9809); stress 0.9741 | **0.965648** |
| sub-05 (not adopted) | + group-level joint assignment (D-018) | 0.994 | 0.944 | 0.9740 (held-out half 0.97429 vs 0.97435) | – |
| sub-06 (not final) | + exact name-only top-5 for empty-address records (D-019), 61 features, t = 0.69 | 0.993 | 0.946 | 0.9745 (India 0.9674, US 0.9816); stress 0.9744 | 0.965155 |

- **Public LB:** final sub-04b **0.965648** (sub-04 0.965297, sub-03 0.960403), vs val macro F0.5 0.9740 (−0.008, the same gap as sub-03 and sub-04). sub-06 had a higher val (0.9745) but LB 0.965155. Differences of this size are within LB noise, so the final is the best-LB version, sub-04b.
- **Portal failures (D-012):** 4 submissions failed evaluation (IDs 3671515, 3671609, 3671650, 3673431). The byte-identical sub-03 file was resubmitted later on 25 Sep and scored, so the failures were on the evaluator side (overload/timeout), not caused by our file or its size. `"source": ""` is not evidence of a missing file; it is empty for every submission, including the scored one. We keep no output size cap (D-012).
- **Test predictions (sub-04b final, t = 0.73):** 5,811,703 matches over 1,631,501 S1, 5.83% predicted singletons (101,043 S1 with no match), file 97.34 MB; 18.25M candidate pairs. Per-country record match rate (best prob ≥ t): France 0.624, India 0.568, US 0.590.
- **France (no labels; sub-04b):** 848,331 test pairs have prob ≥ 0.97 (59.2% of France records, vs India 53.7%, US 57.0%). They cover 244,282 of 259,452 France S1 (94.2%, vs India 93.1%, US 93.9%). 94.9% of France S1 get ≥1 match at t, with 3.64 matches per matched S1 (India 3.51, US 3.59). France's score distribution matches the trained countries, with no sign of a domain-shift collapse.
- **Common false positives (wrong merges):** in sub-02, 18.3% of all predictions were distractors, which is 95% of FPs. Cosine can't separate them (distractor median cos 0.78 vs TP 0.88). Patterns: the same name with an extra token (`holdings`, `group`, `infratech`) or a swapped suffix (inc↔llc), plus a **small house/flat-number shift** (296↔293, flat 106↔101) on an otherwise identical address (90% of distractors). In India there are also different businesses in the same building. At sub-04b, FPs are **0.45% distractor + 0.14% wrong S1** of predictions. Most wrong-S1 errors are empty-address records whose name matches several S1s exactly.
- **Common false negatives (missed matches):** in sub-03, 71% of missed val pairs never reached the matcher (blocking). Of those, Indic-script names (11.4k) and empty addresses (11.8k) were the largest slices. sub-04's dictionary and S1-side retrieval raised India recall from 0.911 to 0.930 (US 0.954 → 0.956). Remaining: empty-address records (only the name is left, and it is often generic) and DBA names recoverable only via the address.

---

## 6. Conclusion
The problem is mostly about precision against engineered distractors, and then about recall for Indic-script and empty-address records. A cheap retrieval stage, features that describe *how* a record differs from its S1 (number drift, extra tokens, suffix swaps, sibling context), and a one-S1-per-record argmax took grouped-holdout macro F0.5 from 0.753 to 0.969. A train-learned Indic token dictionary and S1-side retrieval then raised it to 0.974 (final LB 0.965648) with half the candidates on the val sample (19.4 → 8.6 per S1). A further empty-address name search reached val 0.9745 but not a better LB. Both results held up under a 20% missing-S1 stress test. Lessons: diagnose the FP and FN types before adding model capacity, and audit the candidate generator for ID-order effects, not just the features.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/src/` (entry points run in order; see `README.md`):
- `prep.py`: normalization (`normalize.py`, `translit.py`) and caches for the int64-ID parquet files and GT
- `indic_dict.py`: learns the Indic→Latin token dictionary (train split only) and maps Indic record names
- `blocking.py`: two-stage TF-IDF top-10 per country. `blocking3.py`: blocking v2 union (record-side, S1-side, address-only; `--add-name` adds the empty-address name search). `eval_blocking.py`, `eval_blocking3.py`, `eval_blocking6.py` measure recall and sweep the pruning rules.
- `matcher.py build | train | eval | features | predict | report | submit`: feature parquet files (`features.py`), LightGBM, the val/stress curve, the leakage audit, test inference in record batches, and writing `output/matching_results.tsv` + `output/candidate_pairs.tsv`
- Diagnostics: `diagnose.py`, `errors.py` (FN/FP breakdown), `recovery_diag.py` (blocking-recovery study), `indic_test.py` (Indic retrieval study), `group_assign.py` (group-level assignment, not adopted)
- `check_outputs.py`: runs the official validator. `make_package.py`: builds the submission zip. `metrics.py`: macro F0.5 (reproduces the README example, 0.714).
- Constraints: 15.7 GB RAM, so everything is streamed in chunks per country/part (peak < 12 GB). Only MIT/Apache libraries (LightGBM, scikit-learn, rapidfuzz, sparse_dot_topn). No external data or pretrained models.

### B. Additional Results
- Threshold curve: `work/matcher_curve.tsv` (val and stress P/R/F0.5 for t = 0.05…0.95). Feature importance: `work/matcher_importance.tsv`. Blocking sweep: `work/eval_blocking3.tsv`.
- sub-02 FP breakdown: `work/diagnose_sub02.txt`. Distractor examples: `work/diagnose_examples.txt`. sub-03 error breakdown: `work/errors_sub03.txt`.
