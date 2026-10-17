# Error patterns (feeds template §5)

## Likely false negatives (from EDA of matched train pairs)
- **Script mismatch (India):** S1 is always Latin, but 28% of S2 and 18% of S3 India names are Devanagari (a Latin S1 name against the same words written in Devanagari). 22% of India pairs have name token_set_ratio < 50.
- **Completely different name, same address (DBA/brand):** e.g. a descriptive legal name against an unrelated invented-looking brand word at the same street address. These are recoverable only via the address.
- **Empty address on S2/S3 (~3%)** with a near-identical name. These are recoverable only via the name, and risky for precision.
- **No postal code:** only ~0.3–1% of India/France addresses and ~11% of US addresses contain one.
- **Address component reordering** (`BIG SANDY, MT, 650 ILIAD LOOP`), abbreviations (Rd/Road), zero-padded house numbers (`0274 HIGHWAY 26`), accent noise (`LÍMITED`, `Frânce`).

## Likely false positives (to watch)
- Very generic names (`International International`, `Club SARL`, `SARL`/`SAS`-dominated French names) that share city and street tokens.
- S2/S3 records that are never matched (~26% of records) and look similar to a real match.

## Observed with sub-02 (top-1 + cosine threshold, val)
- **Low precision even at high cosine:** micro-P is only 0.885 at t=0.80 and 0.960 at t=0.90. Likely causes are unmatched distractor records (~26%) whose nearest S1 is a near-duplicate business (same chain/generic name + same city), plus records whose true S1 is outside top-1.
- **India recall ceiling:** R@10 is only 0.941, from Devanagari names (no shared char n-grams with Latin S1) and sparse addresses.

## sub-02 diagnosis on val (t=0.675, `src/diagnose.py`, output in work/diagnose_sub02.txt)
- 1.51M val predictions: TP 80.9%, **FP(a) distractor 18.3%** (the record has no true S1), FP(b1) wrong S1 with the true S1 in top-10 0.6%, FP(b2) wrong S1 with the true S1 not retrieved 0.3%. So 95% of FPs are distractors (India 19.1%, US 17.8%).
- Cosine can't separate them. Distractor cos quantiles (5/25/50/75/95%) are 0.69/0.73/0.78/0.84/0.92, vs TP 0.71/0.79/0.88/0.95/1.00. The gap to rank 2 is similar too (median 0.37 vs 0.45).
- FP(b) sits at cos = 0.707 with gap 0. These are empty-address records tied between S1s with identical names (1/√2 = name cos 1, address 0).
- **Distractors are synthetic near-duplicates of a real S1:** the same name core with a swapped legal suffix (inc↔llc) and/or an extra token (west, group, corporation), plus a different house number on the same street (317 vs 310, 31154 vs 31145). On a 60k sample per country:
  | | all addr numbers equal | exact name equal | legal suffix equal | rec addr empty |
  |---|---|---|---|---|
  | US TP / distractor | 0.73 / 0.02 | 0.37 / 0.006 | 0.73 / 0.35 | 2.8% / 0.1% |
  | India TP / distractor | 0.75 / 0.01 | 0.31 / 0.001 | 0.62 / 0.57 | 2.9% / 0.2% |
  → address-number agreement is the key matcher feature, and exact name / empty address are strong positive signals.
- Leakage check (diagnostic only, not used): corr(row position rec, S1) ≈ +0.003, corr(ID rec, ID S1) ≈ 0. Match rate is flat across file-position and ID deciles (S2 0.733–0.735, S3 0.746–0.747). Median |Δrowpos| is 0.29, which equals the random value 1−1/√2. P(next row has same S1) = 0. **No order/ID leakage.**

## 1a. Distractor patterns (val, t=0.675; `diagnose.py --examples`, full dump in work/diagnose_examples.txt)
| pattern (100k distractor sample) | India | US | All | median cos |
|---|---|---|---|---|
| similar name (token_set ≥ 80), different address numbers | 0.790 | 0.963 | 0.897 | 0.788 |
| different name, similar address (≥ 80) | 0.194 | 0.020 | 0.086 | 0.725 |
| similar name, same address numbers | 0.011 | 0.017 | 0.014 | 0.814 |
| other | 0.005 | 0.000 | 0.002 | 0.712 |
- Typical distractor: the S1 name, often with an added token (`holdings`, `group`, `partners`, `infratech`, `motors`), and the **house/flat number shifted by a small amount** (296↔293, 544↔523, 3751↔3740, flat 106↔101, c 712↔c 711). The rest of the address is identical.
- India "different name, similar address": same building/plot, different business (the same leading place-style name word followed by a different business word).
- True matches at the same cosine (0.75–0.85) also often have "different numbers", but for other reasons: **extra** numbers on one side (PMB 7894, unit/apt, `fl 0`), **zero-padding** (0971↔971, 006367↔6367), or a missing number. The S1 house number is usually still present in the record. TP names also add tokens (`society`, `com`, `aka …`, `formerly …`, `d b a …`).
- → Features: numbers normalized without leading zeros; S1 numbers ⊆ record numbers; count of numbers present only in the record / only in the S1; min |difference| between unmatched numbers (small drift = distractor); extra-name-token count on each side.

## sub-03 val error breakdown (LightGBM matcher, t=0.69; `src/errors.py`, full output in work/errors_sub03.txt)
Val: 200,000 sampled S1, 692,543 GT pairs. TP 645,809, **FN 46,734** (R 0.9325). Predictions to val S1: 650,150, **FP 4,341** (P 0.9933). Macro F0.5 0.9687 (India 0.9580, US 0.9794).

**FN by cause** (share of that country's GT pairs):
| cause | India | US | All |
|---|---|---|---|
| (1) true S1 not in candidates (blocking) | 22,635 (6.5%) | 10,454 (3.0%) | 33,089 (4.8%) |
| (2) in candidates, prob < t | 7,840 (2.3%) | 5,433 (1.6%) | 13,273 (1.9%) |
| (3) assigned to a different S1 | 192 | 180 | 372 |

**FN by slice** (first that applies: Devanagari name → empty record address → name token_set < 50 → other):
| country / cause | Devanagari | empty addr | name_sim<50 (DBA) | other |
|---|---|---|---|---|
| India (1) not in cands | 6,337 | 4,799 | 6,092 | 5,407 |
| India (2) prob < t | 309 | 2,825 | 1,070 | 3,636 |
| US (1) not in cands | 0 | 7,064 | 819 | 2,571 |
| US (2) prob < t | 0 | 2,615 | 673 | 2,145 |

Recall by slice: **empty address India 0.43 / US 0.40** (13.6k / 16.3k pairs), Devanagari India 0.81 (35.2k), DBA India 0.80 / US 0.84, other India 0.965 / US 0.985.

**FP by kind:** distractor 3,386 (India 2,034, US 1,352), wrong S1 955 (India 509, US 446).

**Oracle macro F0.5 gain if one cause is fixed completely** (India / US / all):
| fix | India | US | all |
|---|---|---|---|
| FN (1) blocking misses recovered | +0.0273 | +0.0104 | **+0.0188** |
| FN (2) prob < t recovered | +0.0097 | +0.0060 | +0.0078 |
| FN (3) correctly reassigned | +0.0002 | +0.0002 | +0.0002 |
| FP distractors removed | +0.0049 | +0.0033 | +0.0041 |
| FP wrong S1 removed | +0.0013 | +0.0011 | +0.0012 |

**Takeaways**
- Blocking is the ceiling: 71% of FN never reach the matcher. India misses are split evenly across Devanagari / DBA / empty address / other. US misses are mostly **empty-address records** (7.1k of 10.5k).
- Empty-address records are the worst slice in both countries (R ≈ 0.40–0.43, 37% of all FN). With no address, only the name is left, and generic names have many same-name S1s.
- Wrong-S1 FPs: 11 of the 15 sampled are **empty-address records whose name matches several S1s exactly** (e.g. the same company name registered in two different states). The pick is effectively arbitrary. OCR-style digit-for-letter noise (`h0tels`, `c0`) shows up here too.
- Distractor FPs follow the known pattern: a 1–2 character name typo (swapped or dropped letters in a brand word), an extra token (`company`), a small house-number drift (502/5020, 14/12, 10/8), or a legal-suffix swap (llp/limited).
- **Non-Devanagari Indic scripts occur** (Telugu names were found among the FP samples). The `dev` flag / transliteration may not cover them. This is worth checking for India blocking recall.

**FP examples:** _the 30 record-level examples (15 per kind) were removed from the published version because they reproduced dataset rows._ They showed the patterns summarized above: distractors that differ from the predicted S1 by a 1–2 character name typo, an added or swapped legal word, or a small house/flat-number shift; and wrong-S1 errors that are mostly empty-address records whose name matches several S1s exactly.

## sub-04 audit finding: retrieval tie-break depends on ID order (2026-09-26)
- `matcher.py features` on the v2 candidates: max |Spearman| with record ID 0.168 (n_cand), label vs record ID 0.040. sub-03 had 0.0035 / 0.0009. The effect holds within each country (India: rec~n_cand 0.22, rec~label −0.05, rec~[retrieved S1-side] 0.17).
- Cause: exact cosine ties are common (31% of val pairs tie another candidate). top-k keeps tied items by index position, and records/S1 are indexed in ID order. Pairs found only by the S1-side search are concentrated in the top 4 record-ID deciles (12–19% each vs ~5%).
- Not GT leakage: IDs are not features, and val and test use the same deterministic process. It is an ID-order dependence of the candidate set, though. Fix: index records and S1 in a hash-permuted order so ties break pseudo-randomly; this needs blocking v2 → features → train → predict rebuilt (~3 h).
