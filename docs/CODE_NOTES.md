# Code notes

Function-level comments and docstrings for `code/business_entity_resolution/src/`, grouped by file and function. During the competition the submitted code had to be comment-free, so all commentary was kept here (each file's stripped AST was checked equal to the original). Each module now carries a short module docstring; this file keeps the detailed per-function notes. Entries are in the order they were written, so later sections (e.g. "blocking3.py (sub-04b tie-break fix)") amend earlier ones, and some notes describe the state at that submission (e.g. sub-03 defaults).

## baseline_empty.py

**(module)**
- _(docstring)_ All-empty baseline: every test S1 entity -> no matches, no candidates.
  
  Expected score = test singleton rate (train singleton rate is reported in docs/EDA.md).
  Writes to output/alt/ so it can never overwrite the real submission files in output/.
  Run from repo root: python code/business_entity_resolution/src/baseline_empty.py

## blocking.py

**(module)**
- _(docstring)_ Candidate generation: per country, top-K S1 records for every S2/S3 record.
  
  Features: TF-IDF char 3-grams on name_core + TF-IDF word tokens on address; vocab/IDF fit
  on the country's S1 records only (S1 is Latin-only, so query-only features could never
  match anyway); features with S1 doc freq > max_df dropped. Blocks L2-normalized, weighted
  w_name : 1, hstacked, L2-normalized again -> cosine. Two-stage search: top-N by rare
  features (sparse_dot_topn), rescored with the full vectors -> top-K.
  
  Output: work/blocks/{split}_{country}_K{K}.parquet with columns
    src (int8: 2|3), rec (int64), s1 (int64), rank (int8, 0-based), cos (float32)
  Run: python src/blocking.py --split train --K 10

**topk_rows**
- _(docstring)_ Per-row top-K of (rows, score) pairs; rows must be grouped. Returns (keep idx, rank).

**block_country**
- _(docstring)_ Two stages (D-008): top-N S1 by rare features only (doc freq <= retr_max_df; long
  posting lists dominate matmul cost), then rescore those N with the full vectors -> top-K.

## blocking2.py

**(module)**
- _(docstring)_ Blocking v2 (D-010): union of three per-record retrievals against S1 of the same country.
  
  Views (TF-IDF fit on the country's S1 only):
    n: name_core char 3-grams            (all queries)
    a: address word tokens                (queries with a non-empty address)
    s: consonant skeleton char 2-3-grams  (queries whose name contains Indic script; translit.py)
  Each view: two-stage search (top-N by rare features df <= retr_max_df, rescored with the full
  vectors, D-008) -> top-K per view; union deduped. For every union pair the cosine of all three
  views is computed (cos_c = combined name+address cosine as in blocking v1) and the per-view rank
  is kept (RANK_NONE if the view did not retrieve it).
  
  Output: work/blocks2/{split}_{country}.parquet with columns
    src, rec, s1, cos_n, cos_a, cos_s, cos_c, rank_n, rank_a, rank_s
  Run: python src/blocking2.py --split train            (full)
       python src/blocking2.py --split train --bench 20000   (recall/timing on a query sample)

**View.__init__**
- rows = S1 (for rescoring)
- for retrieval

**View.search**
- _(docstring)_ -> (query row idx, S1 col idx) of the per-query top-K after rescoring.

**block_chunk**
- combined cosine of blocking v1: blocks L2-normalized and hstacked -> (cn + ca) / sqrt(nq * ns)
- S1 addresses are never empty

## check_outputs.py

**(module)**
- _(docstring)_ Pre-submission check: no CR bytes in outputs, then the official validator.
  Run from anywhere: python src/check_outputs.py

## common.py

**(module)**
- _(docstring)_ Shared paths, IO and memory tracking.
- caches; not part of submission

**write_tsv**
- _(docstring)_ The only TSV writer: tab-separated, UTF-8, LF line endings.

**is_val**
- _(docstring)_ Deterministic S1-grouped validation split (~1/frac of S1 entities). s1: int64 array.

**MemTracker**
- _(docstring)_ Samples process private bytes (commit) every 0.1 s; `stage()` reports peak per stage.

## diagnose.py

**(module)**
- _(docstring)_ sub-02 error diagnosis on val (top-1 + cos >= t) and a leakage check on ID / row order.
  
  FP types: (a) record has no true S1 at all (distractor) but was assigned;
            (b) record has a true S1 but was assigned to a different one
                (b1: true S1 inside its top-K list; b2: true S1 not retrieved).
  Leakage check (diagnostic only, never a feature): do ID values or file row positions of
  matched records correlate with their S1's?
  Run: python src/diagnose.py --t 0.675

**main**
- does the record's top-K contain its true S1?
- cos gap to rank 1
- evaluated pairs: predicted S1 in val, OR record's true S1 in val (for FN accounting)
- distractors below threshold too: how many exist among val-assigned rank-0
- ---- leakage diagnostic (ID values and row positions) ----
- are matched records concentrated in some part of the file? (decile match rate)
- ID magnitude vs matched
- adjacent-row check: are consecutive rows of the same S1 (duplicates listed together)?

**_pattern**
- _(docstring)_ Distractor pattern from normalized record/S1 name (core) and address.

**examples**
- _(docstring)_ 1a: 30 distractor FPs and 30 TPs at cos 0.75-0.85 (val), + distractor pattern counts.
- TP needs the right S1, not just any

## eda.py

**(module)**
- _(docstring)_ EDA for the entity-resolution challenge -> docs/EDA.md (stats sections).
  
  Memory-lean: IDs are stored as int64, S2/S3 are streamed in chunks, and pair-level
  stats use a sample of matched S1 entities. Run from repo root:
      python code/business_entity_resolution/src/eda.py [--load-bench]
- \w alone splits Devanagari at vowel signs (category Mn); include combining marks
- ---------------------------------------------------------------- memory helpers

**peak_mem_mb**
- _(docstring)_ (peak working set MB, peak commit MB) of this process (Windows only).

**bench_one**
- _(docstring)_ Child-process entry: load one TSV fully, print JSON stats.

**(module)**
- ---------------------------------------------------------------- helpers
- ---------------------------------------------------------------- main EDA

**main**
- ---------- per-file counts by country, empty rates, postal-like share (streamed)
- split -> src -> Series(id_int -> country)
- (country, field) -> DF
- exact uniform sample of 15: keep the 15 smallest random keys overall
- blocking-scale token document frequencies (test = what we block on)
- ---------- ground truth
- ---------- uniqueness / never-matched
- ---------- country consistency
- ---------- pair sample: fetch names/addresses
- ---------- 7. examples
- ---------- 8. France samples
- ---------- 9. blocking-scale
- pair agreement on keys (train sample). Generic name tokens = top-100 name DF tokens
- of the country (data-driven; train countries derived from train S1+S2+S3 sample pairs).

## errors.py

**(module)**
- _(docstring)_ sub-03 val error breakdown at threshold t (feeds docs/ERRORS.md, template section 5).
  
  FN (GT pair of a sampled val S1 not predicted), by cause:
    (1) true S1 not among the record's candidates, (3) record assigned (prob >= t) to a
    different S1, (2) otherwise: true S1 in candidates but best prob < t.
  Slices (first that applies): Devanagari record name, empty record address, name
  token_set < 50 (DBA), other. FP (prediction to a val S1, wrong): distractor (record has
  no true S1) vs wrong S1. Gain = macro F0.5 with an oracle fix of one cause minus baseline.

**rec_info**
- _(docstring)_ Normalized train S2/S3 fields for record keys (src * 1e12 + rec).

**main**
- candidate presence of the true S1: label==True rows of the val candidate lists
- slices for all GT pairs of val S1
- FPs
- oracle gains (per country and all)
- reassigning removes the wrong-S1 prediction of these records
- examples

## eval_blocking.py

**(module)**
- _(docstring)_ Blocking recall@K on train + top-1/cosine-threshold matcher tuned on the S1-grouped split.
  
  Recall@K = share of GT (record, S1) pairs whose S1 is in the record's top-K list.
  Top-1 matcher: each S2/S3 record -> its rank-0 S1 if cos >= t (D-004: one S1 per record).
  t is chosen on the train part of the split and reported on val.
  Run: python src/eval_blocking.py --K 10

**main**
- per-country val at best t

## eval_blocking2.py

**(module)**
- _(docstring)_ Recall of blocking v2 on train: per view @K, union; overall, per country and slices.
  
  Recall = share of GT (record, S1) pairs whose S1 is among the record's candidates.
  Slices (of GT pairs): indic (record name has Indic script), addr_empty (record address empty),
  name_sim<50 (rapidfuzz token_set_ratio(record name, true S1 name) < 50, normalized names).
  Full mode: denominator = all GT pairs of the country. --bench: GT pairs of queried records.
  Run: python src/eval_blocking2.py [--bench]

**main**
- candidate pairs that are GT pairs, with their per-view ranks
- any view found it
- overall = pair-weighted over countries

## features.py

**(module)**
- _(docstring)_ Pairwise matcher features on blocking-v1 candidates (D-011). Country-agnostic: country is
  only used to partition work (as in blocking), never as a feature.
  
  Candidates: the record's top-10 by combined cosine, pruned to rank < TOPK and (rank 0 or
  cos >= COS_FLOOR). Retrieval/tie/sibling features are computed on the full top-10 list first.
  Sibling features use PREDICTED top-1 assignments only (rank-0 by cosine), never ground truth.
  
  Usage: ctx = Country(split, country); c = ctx.candidates(); for f in ctx.features(c[rows]): ...
- sibling string similarities only for candidates with rank < SIB_RANK

**_nums**
- strips leading zeros; guards int64 overflow

**Entities**
- _(docstring)_ Per-entity text fields in arrays of length n (global positions). With `pos`, only
  those rows are filled (others empty; `loaded` marks filled rows).

**Entities.__init__**
- legal-suffix signature as a shared integer code (0 = none)

**(module)**
- columns produced by Country.candidates() (depend on the candidate lists / top-1 graph)

**Country**
- _(docstring)_ S1 texts are always loaded (TF-IDF fit, small). Record texts are loaded on demand for
  the rows being featurized plus their siblings (load_records), to bound memory.

**Country.__init__**
- name-only / address-only TF-IDF (fit on S1 only, same settings as blocking v1)

**Country.load_records**
- _(docstring)_ Load texts of the records at global positions `pos` (replaces previous load).

**Country.needed_records**
- _(docstring)_ Records of c plus the siblings (rank-0 pointers) of S1s used by sibling features.

**Country.save_graph**
- _(docstring)_ Predicted top-1 table (rank-0 pointer per record) of the last candidates() call.

**Country**
- ---------------------------------------------------------------- candidates

**Country.candidates**
- _(docstring)_ Candidate lists from the blocking-v1 top-K file (optionally without the S1s in
  remove_s1: stress test). Rank, retrieval, tie and sibling-count features are computed
  on the full lists with numpy; only the pruned rows become a DataFrame (memory).
- blocking writes each record's list contiguously, cos descending; verify, else sort
- predicted top-1 assignment graph (rank-0 pointers), never GT
- neighbours within the record's list (full list, before pruning)
- record-level: top-1 vs 2nd
- record-level tie at the top
- this candidate ties another
- rank of this cos among the S1's pointers (other records), 0 = best

**Country**
- ---------------------------------------------------------------- features

**Country.features**
- _(docstring)_ Yields feature frames for consecutive chunks of candidate pairs (bounded memory).
  reuse: pair-level features (key, s1, ...) computed before for the same pairs; only
  missing pairs are recomputed (graph and sibling features are always recomputed).

**Country._pair_feats**
- _(docstring)_ Features that depend only on the (record, S1) pair.
- name/address-only cosines (record TF-IDF only for this chunk's records)
- name strings (core)
- address
- lengths

**_num_feats**
- _(docstring)_ nr/ns: (n, NNUM) int64, -1 padded. Set-style comparisons, vectorized.
- each record number found in S1
- min |diff| between unmatched record numbers and unmatched S1 numbers

## indic_test.py

**(module)**
- _(docstring)_ Step-1 quick test (sub-04 plan): name retrieval for Indic-script records whose true S1 is
  missed by the sub-03 candidates (India). 5k sampled missed pairs; exact top-5 against India S1.
  
    (i)   skeleton: translit.skeleton() on both sides (vowels dropped, aspirates/voicing merged,
          legal words removed); TF-IDF char_wb 2-3 grams fit on S1 skeletons
    (iii) learned token dictionary from TRAIN-split GT pairs (is_val == False) with an Indic
          record name and a Latin S1 name of equal token count, aligned in order; keep
          Indic->Latin maps with count >= 3 and share >= 0.7. Unmapped Indic tokens fall back
          to translit. Searched in two spaces:
          (iii-lat)  mapped name, Latin legal tokens dropped, v1 name vectors (char_wb 3, S1 core)
          (iii-sk)   skeleton(mapped name), same vectors as (i)
    (ii)  multilingual-e5-small: not run (model/torch not on disk, no internet).
  Cost: build + search time, and new (record, S1) pairs per Indic record beyond the pruned v1
  candidates on a random 5k Indic val-record sample. Output: stdout

**main**
- missed Indic pairs among sampled val S1 (as recovery_diag / errors.py cause (1))
- random Indic val-record sample for cost (records GT-matched to a val-split S1 or unmatched)

## make_package.py

**(module)**
- _(docstring)_ Build the submission packages (D-013).
  
    submissions/{tag}_code.zip                     code only (per-submission snapshot, CLAUDE.md)
    submissions/{tag}/{team}_submission.zip        final package, layout from the problem README:
      output/matching_results.tsv, output/candidate_pairs.tsv
      code/business_entity_resolution/{src/, README.md, requirements.txt}
      Documentation_template.md
  No __pycache__ / .pyc. Run: python src/make_package.py sub-03 --team <team_name>

## matcher.py

**(module)**
- _(docstring)_ sub-03 matcher (D-011): LightGBM on pairwise features of blocking-v1 candidates.
  
  Decision: per record take the max-probability candidate; assign it if prob >= t.
  
    build   train split -> work/feats/{fit,val,stress}_{country}.parquet, graph_train_{c}.parquet
            graph  = predicted top-1 table over ALL records (rank-0 pointers; cheap numpy)
            fit    = ~1.5M records with NO val S1 among candidates and true S1 not in val
            val    = N_VAL val S1 per country sampled (seed 42, val_sample_{c}.parquet); records
                     whose candidate list contains a sampled S1 or whose true S1 is sampled
            stress = same records after removing 20% of the sample (seed 43) from all candidate
                     lists; rank/gap/graph/sibling features recomputed, pair features reused
    train   LightGBM (early stopping on a record-grouped 10% of fit, never on val)
    eval    val/stress macro F0.5 per t, per country, FP breakdown (a)/(b)
    predict test -> work/test_pred.parquet (best candidate + prob per record) + candidates
    submit  --t -> output/*.tsv
  Run: python src/matcher.py build|train|eval|features|predict|report|size_t|budget|submit [--t 0.7 --v 0.01]
       build options: --countries India --parts fit val stress (one process per part saves memory)
- val S1 sampled per country
- train S2+S3 records, to split N_FIT across countries

**build**
- val sample of S1 entities of this country + stress removals

**train**
- 10% holdout grouped by record

**val_entities**
- _(docstring)_ Sampled val S1 (all countries, sorted), their country, removed flag; GT S1 column.

**evaluate**
- per country + FP breakdown on val

**check_features**
- _(docstring)_ Pre-predict audit: the model's feature list, no ID / position / GT columns, and each
  feature's |Spearman| correlation with record ID and S1 ID on a val sample.

**predict**
- _(docstring)_ Test: features for all candidates in record batches (texts loaded per batch), best
  candidate per record, + the candidate lists (= candidate_pairs.tsv).
- per-country checkpoint

**report**
- _(docstring)_ Test best-candidate prob distribution and match rate per country (France has no train
  data: a very different distribution would signal a domain shift).

**size_threshold**
- _(docstring)_ Smallest t (plain threshold, v=0) whose matching_results.tsv fits --max-bytes.

**select**
- _(docstring)_ Byte-budget policy (D-012). Among record-level best matches with prob >= t, rank each
  S1's matches by prob; the j-th of n has value prob * (F(j/n) - F((j-1)/n)), F = F0.5 of
  recall at precision 1 (concave: first matches of an S1 are worth most). Keep value >= v.

**out_bytes**
- _(docstring)_ Exact size of matching_results.tsv for S1 order s1_all and matches m (src, rec, s1).

**budget**
- _(docstring)_ Pick v so the test file fits --max-bytes at threshold --t; report val/stress there.

## metrics.py

**(module)**
- _(docstring)_ Per-entity macro F0.5 scorer for the entity-resolution task.
  
  Rules (README "Evaluation Criteria"):
  - F0.5 is computed per Source-1 entity, then averaged over ALL S1 entities.
  - empty truth + empty prediction -> 1.0; empty truth + any prediction -> 0.0.
  - non-empty truth + empty prediction -> 0.0.
- beta = 0.5

**macro_f05**
- _(docstring)_ truth/pred: S1 id -> set of matched ids. Scored over every S1 id in `truth`;
  S1 ids missing from `pred` count as empty predictions.

**macro_f05_counts**
- _(docstring)_ Vectorized macro F0.5 from per-S1 count arrays (one entry per S1 entity).

**_self_test**
- README example: pred 3, truth 2, 2 correct -> P=2/3, R=1 -> 0.714

## normalize.py

**(module)**
- _(docstring)_ Text normalization shared by train and test (hand-written rules, no external data).
  
  norm_name(s)    -> normalized business name
  name_core(n)    -> normalized name with legal-suffix tokens removed (falls back to n)
  norm_address(s) -> normalized address
- Unicode ranges for combining marks we must keep: generic combining diacritics (only
- stripped when their base is Latin) and Indic scripts Devanagari..Sinhala (vowel signs,
- viramas, nuktas are category M and are not matched by \w).
- l.l.c -> llc, s.a.r.l -> sarl
- safety net; dotted forms are collapsed before tokenizing
- Legal-form tokens (after NAME_MAP), incl. Devanagari spellings seen in India S2/S3.

**_AccentTable**
- _(docstring)_ str.translate table built lazily per character.
  - Indic script chars (U+0900-U+0DFF): kept as-is (vowel signs survive).
  - standalone combining diacritics U+0300-U+036F: dropped (they only follow Latin here).
  - other non-ASCII chars: NFKD, drop combining marks (é->e, º->o, Ç->c, ﬁ->fi).

**name_core**
- _(docstring)_ n must already be normalized with norm_name.

**norm_address**
- "n° 25" / "nº 25" become "n 25" or "no 25" after punctuation removal

**normalize_df**
- _(docstring)_ Adds name, core, addr columns (python lists -> fast; pandas .map is slower).

## prep.py

**(module)**
- _(docstring)_ Stream raw TSVs -> normalized parquet cache work/norm/{split}_{src}.parquet.
  
  Columns: i (int64 id without prefix), country, name, core, addr, dev (Devanagari in raw
  name), addr_empty. Train also writes train_gt.parquet (see prep_gt).
  Run: python src/prep.py --split train|test [--gt-only]

**prep**
- int64 ids must round-trip (no leading zeros), else keep a string lookup

**prep_gt**
- _(docstring)_ GT -> work/norm/train_gt.parquet: one row per matched record (src int8, rec, s1).
  Each S2/S3 record has at most one S1 (D-004), so (src, rec) is a unique key.

## recovery_diag.py

**(module)**
- _(docstring)_ Diagnostic only (no pipeline change): which extra retrievals would recover the val GT pairs
  whose true S1 is NOT in the sub-03 candidates (cause (1) in errors.py)?
  
  Per country, vectorizers as blocking v1 (fit on the country's S1: name_core char_wb 3-grams,
  address word tokens, max_df 0.05, each block L2-normalized; combined = normalize(hstack)).
  All searches are exact (sparse_dot_topn on full vectors, no rare-feature stage; see D-010).
    (a) name-only, record -> S1 top-5
    (b) address-only, record -> S1 top-5
    (c) S1-side: each S1 -> its top-10 train records (S2+S3 of the country) by combined cosine
    (d) sibling: another record with name-only or address-only cos >= 0.9 whose v1 top-1 S1 is the S1
    (e) name-only top-5 on translit(core) (all Indic scripts incl. Telugu/Kannada)
  Extra candidates: on 20k random val records (10k random S1 for (c)), the mean number of new
  (record, S1) pairs per record that are not already in the pruned v1 candidates (D-011).
  Output: stdout tables + work/recovery_diag.tsv

**topn_pairs**
- _(docstring)_ rows of A, cols of B (BT = B.T as csr), score; chunked for progress / memory.

**existing**
- _(docstring)_ Pruned v1 candidates (rank < 5 and (rank 0 or cos >= 0.3), D-011) + v1 top-1 per record.

**main**
- S1 vectors
- record vectors (all train records of the country)
- query records: missed + random val-record sample
- method -> DataFrame(key, s1)
- (d) siblings -> their v1 top-1 S1
- (c) S1-side top-10 records, for the missed pairs' S1 + a random S1 sample
- recovery on missed pairs
- extra candidates per record (beyond pruned v1 candidates)

## size_threshold.py

**(module)**
- _(docstring)_ Pick the top-1 cosine threshold under a matching_results.tsv size cap (D-009).
  
  For each t: exact byte size of matching_results.tsv on test (from rank-0 test pairs),
  val macro F0.5 (normal), and val macro F0.5 under a distractor stress test: 20% of val S1
  entities (seed 42) are removed from the candidate lists, so each record falls back to its
  best remaining top-K candidate, and records of removed S1s become distractors.
  Run: python src/size_threshold.py --max-bytes 90000000

**test_sizes**
- "S1-" + id + tab + newline
- commas = n - 1 per row

**val_scores**
- stress: drop removed S1s; rows are rank-sorted within record -> first = best remaining

## submit_top1.py

**(module)**
- _(docstring)_ sub-02: top-1 S1 per S2/S3 record + cosine threshold -> output/*.tsv for the test split.
  
  candidate_pairs.tsv = rank-0 retrieval pairs (exactly what the threshold "model" scores).
  matching_results.tsv = candidates with cos >= t (t tuned by eval_blocking.py on train).
  Run: python src/submit_top1.py --t 0.6 --K 10

**join_ids**
- _(docstring)_ Comma-joined 'S{src}-{rec}' lists aligned to s1_order ('' when none).

## test_normalize.py

**(module)**
- _(docstring)_ Unit tests for normalize.py on examples from docs/EDA.md. Run: python src/test_normalize.py

**test_india**
- Devanagari vowel signs survive; legal words removed from core
- Telugu marks kept

**test_france**
- fallback when core is empty

## translit.py

**(module)**
- _(docstring)_ Hand-written Indic -> Latin transliteration and a script-neutral consonant skeleton.
  
  Devanagari table below; the other Brahmic blocks (Bengali..Malayalam, incl. Telugu +0x300
  and Kannada +0x380) follow the same ISCII-derived layout, so offsets are reused.
  translit("कमल ट्रेडर्स")  -> "kml tredrs" (invented example)
  skeleton(...)            -> "kml trtrs" (aspirates and voicing merged, c/q->k, vowels dropped)
- offset within a 0x80 block -> Latin
- virama
- Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam
- Malayalam chillu
- Tamil ca: mostly "s" in loanwords (sun, systems)
- Malayalam rra+virama+rra = "tt"
- Tamil aytham + pa = f
- voicing merged (Tamil script does not mark it): d->t, b->p, g->k
- keep a word-initial vowel

**translit**
- _(docstring)_ Indic characters -> Latin; everything else unchanged. s should be normalized.

**_skel**
- placeholder so the later c->k does not touch them

**skeleton**
- _(docstring)_ Script-neutral consonant skeleton of a normalized name (Latin or Indic), legal words
  removed (falls back to the full skeleton if nothing else is left).

## indic_dict.py

**(module)**
- _(D-014)_ Learns an Indic-token → Latin-token dictionary from TRAIN-split GT pairs only (is_val false): Indic-script record name vs Latin S1 name with equal token counts, aligned in order. Keeps maps with count ≥ 3 and share ≥ 0.7. Saved to work/indic_dict.json.
- Then rewrites `core` of every S2/S3 record whose name contains Indic script (train and test) to the mapped name. Unmapped tokens fall back to translit and Latin legal words are dropped. The original core is kept in `core_raw`. The step is idempotent: `core` is always rebuilt from `core_raw` and `name`.
- No country is hard-coded; Indic names are detected by script.
- Run after prep.py, before blocking: `python src/indic_dict.py`

## blocking3.py

**(module)**
- Blocking v2 (sub-04, D-015): per country, the union of three retrievals, all in the v1 TF-IDF spaces fit on the country's S1 (name char_wb 3-grams + address words, max_df 0.05; retrieval stage on features with S1 df ≤ 5000, N=50, rescored with the full vectors):
  - `rank_v1`: v1 record→S1 top-10. Reused from work/blocks for non-Indic records; recomputed for records with an Indic-script name, whose `core` is now the dictionary-mapped name (D-014).
  - `rank_a`: address-only record→S1 top-3 (records with a non-empty address).
  - `rank_s`: S1-side, S1→record top-10 by combined cosine.
- The union is keyed by a single int64 pair id (record position × #S1 + S1 position), deduped with np.unique; per-source ranks are kept (127 = not retrieved by that source). `cos` is the combined v1 cosine for every pair. Output: work/blocks3/{split}_{country}{tag}.parquet sorted by record, cos desc.
- `--s1-limit N` runs the S1-side query for a random N-S1 sample only (bench).

## eval_blocking3.py

**(module)**
- Sweeps the pruning rule on train: keep a pair if (record rank < K and (rank 0 or cos ≥ top1 − delta)) or S1-side rank < ks or address rank < ka. Recall is over GT pairs of the S1s that were queried S1-side; cost is kept pairs per S1. The baseline is the sub-03 rule on the old v1 lists (rank < 5 and (rank 0 or cos ≥ 0.3)).

## features.py (sub-04 additions)

**(module)**
- `BLOCKS` (env `ER_BLOCKS`, default `v3`) picks the candidate source: `v3` = work/blocks3 union (blocking3.py, D-015), `v1` = the sub-03 lists in work/blocks.
- `V3_DELTA, V3_K, V3_KS, V3_KA`: the v3 pruning rule. Keep a pair if (record rank < K and (rank 0 or top1 cos − cos ≤ delta)) or S1-side rank < KS or address rank < KA. Values come from the eval_blocking3.py sweep on train.
- `SRC_COLS` (rank_v1, rank_a, rank_s; 127 = not retrieved by that source) are carried into the candidates as graph features, so the matcher knows which retrieval found each pair.

**Country.candidates**
- `n_cand` is capped at 127 (int8). v3 records can have more than 127 union pairs, because many S1s can pull the same record S1-side.
- Sibling linking (d) is not a separate retrieval. The existing `sib_max_name` / `sib_max_addr` features (max similarity to other records whose top-1 is this S1) already encode it.

## blocking3.py (sub-04b tie-break fix)

**mix64 / hperm / two_stage_perm**
- Exact cosine ties are common (identical normalized records). sparse_dot_topn + topk_rows keep tied items by index position, and the S1/record indexes were in ID order, which made the candidate set depend on ID order (ERRORS.md, sub-04 audit). `hperm` orders each index by a splitmix64 hash of the ID; `two_stage_perm` searches the permuted index and maps the columns back, so ties break pseudo-randomly and deterministically.

**block_country**
- The final per-record sort breaks cos ties by mix64(key) ^ mix64(s1), not by S1 position, so the record-rank pruning in features.candidates is also ID-independent.
- The non-Indic v1 lists are still reused from blocking.py. Their tie effect was negligible in the sub-03 audit (|Spearman| ≤ 0.006); the matcher.py features audit on sub-04b re-checks this.

## matcher.py (sub-05 prep)

**top2**
- Keeps each record's best AND second-best distinct S1 with probabilities (`s1_2`, `prob_2`; −1 / 0 when a record has one candidate). It is used by `best_per_record` (best_val / best_stress) and by `predict` (test_pred_*.parquet). Existing readers use only the original columns. The second-best is what sub-05's group assignment re-routes rejected members to.

## group_assign.py (sub-05, D-018, not adopted)

**(module)**
- Group-level joint assignment on top of the matcher's per-record best/second-best S1. `assign`: members = records with prob ≥ floor, grouped by top-1 S1. Accept a group if mean prob ≥ T[size bucket]. Members of rejected groups stay if prob ≥ solo; otherwise they go to their second-best S1 if that group's mean with them added clears T. `baseline(t)` (floor t, solo off, T = 0) reproduces per-record argmax exactly.
- `tune`: coordinate descent on a hash half of the val S1; the other half is reported as held-out. `eval` prints val/stress per half, per country and per slice; `submit` writes matching_results.tsv from test_pred.parquet with the saved params.

## blocking3.py (sub-06 add-on)

**add_name**
- `--add-name`: for records whose normalized address is empty (the weakest blocking slice, recall ≈ 0.71), runs an exact name-only top-5 search against the country's S1 with the v1 name vectors (char_wb 3-grams, max_df 0.05). There is no two-stage search, because empty-address names have too few rare features, and the S1 index is hash-permuted for tie-breaking. The new pairs are merged into the existing blocks3 union with `rank_n` (127 = not from this search) and their combined cosine. Records are re-sorted by cos with the pair-hash tie order. Re-running is idempotent: pairs found only by a previous name search are dropped first.

## features.py (sub-06)
- `V3_KN` (env `ER_KN`, default 0 = off, which reproduces the final sub-04b; 5 = the sub-06 attempt): also keep pairs with name-only rank < K_n. When on, `rank_n` becomes a graph feature.

## eval_blocking6.py
- Sweeps K_n ∈ {0, 1, 3, 5} on top of the chosen v2 rule, on the val-sample S1. Reports overall and empty-address pair recall, candidates per S1, and extra kept pairs per empty-address record.

## export_texts.py (sub-07)

**(module)**
- Runs in `.venv`, because `.venv_gpu` has no pandas/pyarrow and installing them would need internet. For each split/country it writes `work/embeddings/{split}_{country}_{S1|R}.txt` (one text per line: normalized name + " " + normalized address, lowercased, whitespace-collapsed; the raw Indic script is kept) plus `_keys.npy` (S1 ids; record keys src·10¹² + id), sorted by key.

## embed.py (sub-07, runs in .venv_gpu)

**(module)**
- Encodes each shard with multilingual-e5-small, loaded offline from the local HF cache (HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE, local_files_only). Uses fp16, batch 1024, max_seq_length 128 (sampled texts are ≤ 65 tokens, so nothing is truncated), and the e5 "query: " prefix on both sides (e5's recommendation for symmetric similarity). Embeddings are L2-normalized and written as float16 into a memory-mapped `work/embeddings/{split}_{country}_{S1|R}.npy`, row order = `_keys.npy`. A `.done` marker makes it resumable. Prints throughput every 10k records.
- `--limit N`: timed test on the first N records of the first shard. The measured result was ~3,800–4,000 rec/s steady, 0.77 GiB peak VRAM on the RTX 3050 6 GB.

## dense_search.py (sub-07, runs in .venv_gpu)

**(module)**
- Exact dense top-10 per record against the same country's S1: fp16 matrix product on the GPU in record chunks (4,096) × S1 blocks (131,072), top-k per block, then a merged top-k. The S1 order is hash-permuted (splitmix64, same as blocking3.py) so exact ties never follow ID order. Output `work/dense/{split}_{country}_pairs.npz` (record_id, s1_id, cos_d, rank_d); npz rather than parquet because there is no pyarrow in `.venv_gpu`.
