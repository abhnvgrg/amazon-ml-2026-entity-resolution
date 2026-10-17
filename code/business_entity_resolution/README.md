# Business Entity Resolution — Amazon ML Challenge 2026

Python 3.14, `pip install -r requirements.txt`. Run from the repo root (it expects `dataset/`
next to `code/`). Caches go to `work/`, outputs to `output/`. No internet or external data is used.
The competition data and the competition's `utils/validate_submission.py` are not included in this
repository; place them at those paths to run the full pipeline and the validation step below.
`requirements-dense.txt` (Python 3.12, GPU) is only for the abandoned sub-07 e5 experiment.

Final pipeline (sub-04b: Indic name dictionary → blocking v2 with ID-independent tie-breaking → LightGBM pairwise matcher, D-014–D-017):

```
S=code/business_entity_resolution/src
python $S/prep.py --split train
python $S/prep.py --split test
python $S/indic_dict.py
python $S/blocking.py --split train --K 10
python $S/blocking.py --split test --K 10
python $S/blocking3.py --split train
python $S/blocking3.py --split test
python $S/matcher.py build
python $S/matcher.py train
python $S/matcher.py eval
python $S/matcher.py features
python $S/matcher.py predict --batch 500000
python $S/matcher.py report --ts 0.5 0.73 0.9
python $S/matcher.py submit --t 0.73
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
python $S/make_package.py sub-04b --team "git push --force"
```

Steps:

1. `prep.py` normalizes names and addresses into `work/norm/` and caches the train ground truth (~3 min per split).
2. `indic_dict.py` learns an Indic-token → Latin-token dictionary from train-split matched pairs only, then rewrites the name core of every Indic-script S2/S3 record (train and test). The original core is kept as `core_raw`.
3. `blocking.py` is the v1 record → S1 TF-IDF top-10 (~45 min per split, ~6 GB peak). `blocking3.py` reuses it and adds the S1-side top-10 and address-only top-3 retrievals, writing the union to `work/blocks3/` (~65 min per split, ~12 GB peak).
4. `matcher.py build/train/eval` builds pair features on the pruned v2 candidates, trains LightGBM and picks the threshold on val (t = 0.73). `features` audits that no ID, position or ground-truth column is a model feature.
5. `predict` scores all test candidates (~1.5 h, <10 GB peak). `submit` writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`, with no output size cap (D-012).
6. `make_package.py` builds `submissions/sub-04b/<team>_submission.zip` in the layout required by the problem statement.

`candidate_pairs.tsv` holds exactly the pruned candidate lists the matcher scored. The pruning rule keeps a pair if (record rank < 10 and (rank 0 or top-1 cos − cos ≤ 0.1)) or S1-side rank < 5. Setting `ER_BLOCKS=v1` reproduces sub-03 (v1 top-5 lists, t = 0.69). The sub-06 attempt (D-019, LB 0.965155, below sub-04b's 0.965648) adds `python $S/blocking3.py --split train|test --add-name` (exact name-only top-5 for empty-address records) and runs with `ER_KN=5` and t = 0.69. Retrieval indexes are hash-permuted, so exact-cosine ties never depend on ID order. The sub-02 scripts (`submit_top1.py`, `size_threshold.py`) and the diagnostics (`diagnose.py`, `errors.py`, `recovery_diag.py`, `indic_test.py`, `eval_blocking*.py`) and the rejected group assignment (`group_assign.py`) are kept for reference.
