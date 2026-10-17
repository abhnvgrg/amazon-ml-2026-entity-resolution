# Decisions

## D-001 [2.1 EDA] Memory-lean data handling
- Options: load everything with pandas str columns; stream in chunks and store IDs as int64 / country as categorical.
- Choice: stream + int64 IDs (strip `S?-` prefix; source kept separately) + categorical country.
- Evidence: 15.7 GB RAM, **no pagefile**, ~6.7 GB commit free with apps closed. One S2/S3 file costs ~1.45 GB of peak commit as pandas str (no pyarrow), so all 3 test sources come to ~3.4 GB. Full EDA with streaming peaked at 2.3 GB. Also recommended: enable a pagefile before building the full pipeline.

## D-002 [4 Threshold] Metric implementation
- `src/metrics.py`: per-S1 F0.5, macro over all S1. Empty/empty = 1, empty truth + any pred = 0. The README example reproduces 0.714.
- Also reports micro P/R to support threshold tuning.

## D-003 [3 Blocking] Block within country; postal codes are not a primary key
- Evidence: 0 of 7.64M matched train pairs cross countries, so it is safe to partition by the `country` string (open set, not hard-coded).
- Postal-like 5–6 digit numbers appear in only ~11% of US addresses and ≤1% of India/France addresses. Only ~8% of US pairs and 0.3% of India pairs share one, so postal-code blocking is useless as a primary key.
- Name/address-token and address-number keys are better candidates. On sampled pairs, first non-generic name token OR shared address number covers ≥95%; see EDA §9.

## D-004 [4 Features/Model] Exploit one-S1-per-record constraint
- Evidence: 0 S2/S3 IDs appear under more than one S1 in train GT, and ~26% of S2/S3 records are never matched (distractors).
- Plan: at inference, assign each S2/S3 record to at most one S1 (best-scoring), which is a cheap precision gain.

## D-005 [2.1 EDA] Tokenizer must keep Devanagari combining marks
- `\w+` splits Hindi words at vowel signs (`प्राइवेट` came out as 4 fragments), so use `[\ẁ-ͯऀ-ॿ]+`.
- Evidence: 28% of India S2 names and 18% of India S3 names are non-ASCII (Devanagari), while S1 India is 0%. Matching them needs transliteration. Train pairs give aligned Latin/Devanagari names that can be learned from (no external data).

## D-006 [3 Blocking] Per-country TF-IDF top-K retrieval (S2/S3 → S1)
- Options: token/number key blocking (D-003); sorted neighbourhood; TF-IDF cosine top-K retrieval.
- Choice: fit per country on S1 only. Features are char-3-gram TF-IDF on the name core (legal suffixes removed) plus word TF-IDF on the address, max_df 5%, L2-normalized per block, then hstacked and normalized again. Each S2/S3 record retrieves its top-K S1 (sparse_dot_topn). Direction is record→S1 because each record has ≤1 S1 (D-004).
- Evidence: see EXPERIMENTS (recall@K on train). The 20k-S1 sample gives recall@1 0.987 and @10 0.996. The sample is inflated because it has fewer competing S1s.

## D-007 [4 Threshold] sub-02 matcher = top-1 S1 + global cosine threshold
- Choice: assign each record to its rank-0 S1 if cos ≥ t. t is tuned on the train part of the S1-grouped split (`common.is_val`, hash-based, 20% val) and reported on val. One global t, with no per-country thresholds, because France has no train data.
- candidate_pairs.tsv = rank-0 pairs (the exact set the threshold scores).

## D-008 [3 Blocking] Two-stage retrieval for speed (rare-feature top-200 → full rescoring)
- Problem: exact top-K with max_df=5% took about 32 s per 20k queries on India (883k S1), so a full train run would take hours. Frequent features (long posting lists) dominate the sparse matmul cost.
- Options (India, 20k S2 queries): cut max_df to 5000 → R@1 0.851 (exact 0.918). Or retrieve top-200 with df≤5000 features, then rescore with the full vectors: R@1 0.915, R@10 0.946 (exact 0.955), 3.4 s per 20k.
- Choice: two-stage, N=200, retr max_df=5000 (absolute), 20k-query chunks (rescoring gathers 4M rows; ~3.5 GB peak).

## D-009 [4 Threshold] sub-02b: raise t to fit a suspected ~100 MB portal size limit
- Problem: the portal rejects the 112.8 MB (107.6 MiB) sub-02 matching_results.tsv even though its format is clean. Suspected evaluator size limit ~100 MB (unconfirmed).
- Options: keep t=0.675 (best val F0.5 0.753, 112.8 MB); raise t until the file is ≤ 90×10⁶ bytes (10% margin under 100 MB). `src/size_threshold.py` computes the exact size per t from the rank-0 test pairs.
- Choice: t=0.765, the lowest t that fits: 89.89 MB, 5.23M matches. Val P 0.859 / R 0.650 / macro F0.5 0.733 (−0.020). Stress (20% of val S1 removed, seed 42) gives 0.733 as well, so the top-1 matcher is insensitive to missing S1s.
- Note: cos=1/√2≈0.707 is a cliff (exact name + empty address); t>0.707 drops all such records (−0.014 F0.5). A learned matcher should recover them within the size budget.

## D-010 [3 Blocking] Union-of-views blocking deferred to sub-04 (bench findings)
- Tried: blocking2.py, a union of name-only / address-only / skeleton retrievals (two-stage). 20k-query bench, union recall 0.951 at 17.7 cand/record (India 0.933, US 0.970). That's below v1 combined R@10 0.959 at 10 cand.
- Findings: exact (not two-stage) name-only char-3 search reaches only R@1 0.43 / R@10 0.60 on India. Many S1 share name words, so name alone can't place the right S1 in the top 10. Rare-feature two-stage retrieval breaks down for name and skeleton views: nearly all char n-grams have df > 5000 on 883k S1, so the retrieval stage has almost no features. Name words + rare n-grams (df≤20000, N=200) reach exact recall at half the cost.
- Choice: sub-03 keeps the v1 top-10 candidates, since 95% of FPs are distractors (precision problem, not recall). Revisit in sub-04 with word-based retrieval for the name/skeleton views. translit.py (hand-written Indic→Latin + consonant skeleton; true-pair skeleton sim median 80 vs 32 random) is reused as a matcher feature.

## D-011 [4 Features/Model] sub-03 matcher: LightGBM on v1 top-10 candidates
- Candidates: v1 combined-cosine top-10, pruned to rank < 5 and (rank 0 or cos ≥ 0.3). That's ~4.2 pairs/record instead of 10, and loses 0.7% (India) / 0.2% (US) of true pairs, all at cos < 0.3.
- Features (`features.py`, no country feature):
  - retrieval: cos, rank, gap to top, gap to next, top1–top2 gap, tie flags
  - name/address disagreement: name-only and address-only TF-IDF cos, diff, product, hi/lo flags
  - name: token_set/sort, partial, Jaro-Winkler, full-name ratio, translit skeleton ratio, exact name/core, extra tokens per side, legal-suffix agreement, Indic flag
  - address: token_set, non-numeric token_set, numbers without leading zeros (first equal, all equal, S1 ⊆ record, shared, record-only, S1-only, log min |diff| of unmatched, log first diff), empty flag
  - sibling features from **predicted** rank-0 assignments only: in-degree S2/S3, other-source sibling, cos rank among pointers, cos vs S1's best pointer, max name/address token_set to siblings for rank < 2
  - lengths/token counts, source
- Data: fit = 1.5M records with no val S1 among candidates and true S1 not in val; val = records with any val-S1 candidate or val true S1 (exact attribution for val S1). Early stopping on a record-grouped 10% of fit, never on val.
- Decision: argmax prob per record; assign if prob ≥ t.

## D-012 [4 Threshold] No output size cap; supersedes D-009
- Options: keep the ≤ 90 MB cap from D-009 (a higher t or a byte-budget cut, which costs F0.5); drop the cap and submit the val-optimal output.
- Choice: no size cap. Primary output = sub-03 matcher at t=0.69 with no budget cut (val macro F0.5 0.9687, stress 0.9687).
- Evidence: 4 portal submissions failed evaluation (IDs 3671515, 3671609, 3671650, 3673431). The byte-identical sub-03 file (96,690,609 bytes) was resubmitted later on 25 Sep and scored 0.960, so the failures were on the evaluator side (overload/timeout), not caused by our file or its size. The ~100 MB limit in D-009 is unfounded.
- Note: `"source": ""` is not evidence of a missing file. It is empty for every submission, including the one that was scored.

## D-013 [A Code Artefacts] Final package layout per problem README
- Options: code-only zip (sub-NN_code.zip, as before); the README's `<team_name>_submission.zip` with output/, code/business_entity_resolution/{src, README.md, requirements.txt} and Documentation_template.md.
- Choice: make_package.py builds both: submissions/{tag}_code.zip (per-submission snapshot) and submissions/{tag}/{team}_submission.zip (final). Team name "git push --force".
- Evidence: sub-03 test build has 26 files, 290,247,588 bytes. The zip passes testzip, and the zipped matching_results.tsv has the same sha256 (d991fc90…8941) as the scored file.

## D-014 [3 Blocking] Indic names: learned token dictionary, searched with the address (sub-04)
- Tried on 5k sampled val pairs with an Indic-script name whose true S1 was missed (11,389 total), exact top-5 vs India S1 (`src/indic_test.py`): (i) consonant skeleton 0.087; (iii) learned dictionary, name-only 0.204 (Latin char-3) / 0.184 (skeleton space); **(iii) dictionary name + address in the v1 combined space 0.942**. References: translit char-3 0.022, v1 combined with the raw Indic name 0.076. (ii) multilingual-e5-small was not run: the model and torch are not on disk, and there is no internet.
- Dictionary: from TRAIN-split GT pairs only (is_val false), Indic record name vs Latin S1 name with equal token counts, aligned in order. 418,920 pairs, 1,312 Indic token types, 1,281 kept (count ≥ 3, share ≥ 0.7). Coverage of Indic token occurrences: val-split records 95.5%, TEST India records 91.7% (867,245 Indic records). Unmapped tokens fall back to translit.
- Choice: adopt (iii) with the address. Name-only search can't separate the many S1s that share generic name words (as in D-010). The v1 name vector for an Indic name is empty (no shared n-grams with the Latin S1 vocabulary), so the mapped name fills exactly that gap. Cost: ~4.0 new pairs per Indic record if added as a separate view, ~30 min search for all test Indic records.

## D-015 [3 Blocking] Blocking v2: v1 ∪ S1-side top-10 ∪ address top-3, adaptive pruning (sub-04)
- Built (`blocking3.py`): v1 record→S1 top-10, with Indic records re-queried on the dictionary-mapped name (D-014); ∪ address-only top-3; ∪ S1-side S1→record top-10. All two-stage (rare features df ≤ 5000, N=50, rescored). Train: India 49.8M, US 74.5M union pairs. S1-side costs 0.9 ms/S1 (exact search took ~12 ms/S1). Sibling linking (d) is not a retrieval; the existing sib_max_name/sib_max_addr features already encode it.
- Pruning swept on the 200k val-sample S1 (`eval_blocking3.py`): keep if (record rank < K and (rank 0 or top1 cos − cos ≤ delta)) or S1-side rank < ks or address rank < ka.
- Choice: delta 0.1, K 10, ks 5, ka 0. Pair recall 0.9653 (India 0.9552, US 0.9755) at 8.5 candidates/S1, vs sub-03 0.9524 (0.9349 / 0.9699) at 19.4. The unpruned union reaches 0.9666 at 55.9. The address view adds ≤ 0.0003 once S1-side is in, so ka = 0; its rank stays a feature. Most of the India gain comes from the Indic mapping (blocking recall for Indic-script names ≈ 0.82 in sub-03 (Devanagari slice, 1 − 6,337/35,159) → 0.91 on the v2 bench). Empty-address records remain the weakest slice (0.71): their names have few rare features for the two-stage search.

## D-016 [4 Features/Model] sub-04 matcher: LightGBM retrained on blocking v2 candidates
- Same features and params as D-011, plus the retrieval-source ranks (rank_v1, rank_a, rank_s): 60 features. Fit 1.82M pairs / 1.5M records (≈1.2 pairs/record after adaptive pruning), best iteration 1789. rank_v1 is the 5th feature by gain (6.8%).
- Val/stress (same 200k val-sample S1 and the same 20% stress removal as sub-03): best t = 0.71, val macro F0.5 **0.9737** (P 0.9938, R 0.9433), stress **0.9737**. sub-03 had 0.9687 / 0.9687. Per country: India 0.9580 → 0.9667, US 0.9794 → 0.9807. FP mix: distractor 0.47%, wrong S1 0.16%.
- Choice: adopt (both val and stress beat 0.9687, as the plan required). Primary output = sub-04 at t = 0.71. The flat curve (0.9733–0.9736 for t in 0.60–0.80) makes the threshold robust. sub-03 artifacts are kept in work/sub03/.

## D-017 [3 Blocking] sub-04b: ID-independent tie-breaking in blocking v2
- Problem: the sub-04 audit showed the candidate set depended on ID order (|Spearman| rec ID vs n_cand 0.168, label vs rec ID 0.040; ERRORS.md). Exact-cosine ties were broken by index position.
- Fix: blocking3.py searches hash-permuted (splitmix64) S1/record indexes, maps results back, and breaks per-record cos ties by a pair hash. Pruning, features and model params are unchanged; blocking v2, features, training, eval and predict were rebuilt.
- Result: audit clean again (max |Spearman| 0.0041 rec ID, 0.0070 S1 ID; label 0.0011 / 0.0001). Val macro F0.5 0.9740 (P 0.9942, R 0.9430; India 0.9672, US 0.9809), stress 0.9741, at t = 0.73. sub-04 had 0.9737 / 0.9737. Both beat sub-04, so sub-04b replaces sub-04. Validator PASS with --check-ids.

## D-018 [4 Threshold] sub-05 group-level assignment: not adopted
- Tried (`group_assign.py`, on sub-04b val/stress scores incl. second-best S1): group records by predicted top-1 S1 (prob ≥ floor). Accept a group if its mean prob ≥ a size-dependent threshold T[size ∈ {1, 2, 3–4, 5–6, 7+}]. Members of rejected groups stay if prob ≥ t_solo, otherwise try their second-best S1 (accepted if that group, with them added, clears T). Coordinate descent over floor, t_solo and the 5 T values, tuned on a hash half (A) of the val S1 and reported on the other half (B).
- Result: no group, solo or re-routing setting beat per-record argmax on A. The tuned rule collapsed to "floor 0.71", i.e. a plain per-record threshold: A 0.97376 vs 0.97374, B 0.97429 vs **0.97435**, val 0.97403 vs **0.97405**, stress 0.97406 vs **0.97409**. It trades precision for recall in every slice (e.g. empty address P 0.950→0.945, R 0.494→0.498).
- Choice: keep sub-04b (per-record argmax, t = 0.73). The matcher's graph/sibling features (in-degree, sibling similarity) already encode the group context, so a group mean adds nothing. This matches the sub-03 oracle bound: fixing every wrong-S1 error was worth only +0.0012.

## D-019 [3 Blocking] sub-06: exact name-only top-5 for empty-address records (rank_n, K_n = 5)
- Built (`blocking3.py --add-name`): for records with an empty normalized address (train: India 122,794, US 222,089; test: France 43,078, India 112,004, US 110,424), an exact name-only top-5 against S1 (v1 name vectors, hash-permuted index), merged into the v2 union as `rank_n`. Kept if rank_n < K_n; rank_n is also a matcher feature (61 features).
- K_n sweep on the val-sample S1 (`eval_blocking6.py`): empty-address blocking recall 0.7684 → 0.7690 / 0.7702 / **0.7722** at K_n = 1 / 3 / 5, with 0.41 / 1.16 / **1.65** extra candidates per empty-address record. Overall pair recall 0.9701 → 0.9703, candidates/S1 8.61 → 8.86. Most Part-B name-only recoveries were already covered by S1-side retrieval once ties were hash-broken.
- Gate (sub-04b model, sub-06 candidates, t = 0.73): val 0.97448 / stress 0.97445 vs sub-04b 0.97405 / 0.97409. Adding the name-only candidates also changes rank/gap/n_cand for those records, so recall gained more (+0.0009) than blocking recall alone.
- Retrained (same hyperparameters, 61 features, best iteration 1821): val-optimal t = 0.69, val **0.97453** (P 0.9933, R 0.9464; India 0.9674, US 0.9816), stress **0.97442**. The curve is flat (≥ 0.9744 for t 0.66–0.75). Audit clean (max |Spearman| 0.0045 rec ID, 0.0067 S1 ID; label 0.0004 / 0.0011).
- Choice: passed the val/stress gate, but public LB 0.965155 < sub-04b 0.965648. The val gain (+0.0005) did not transfer; the LB difference is within noise. **sub-04b is the final submission.** The add-on stays in code, off by default (ER_KN=0); ER_KN=5 with `--add-name` blocks reproduces sub-06.
