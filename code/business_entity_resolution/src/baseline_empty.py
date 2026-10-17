"""All-empty baseline (sub-01).

Writes every test S1 entity with no matches and no candidates to output/alt/, so it can
never overwrite the real submission files. Its score equals the test singleton rate
(0.056 on train). Kept as a format/validator smoke test.
"""

import pandas as pd

from common import DATA, OUT, write_tsv


def main() -> None:
    ids = pd.read_csv(DATA / "test" / "test_source1.tsv", sep="\t", dtype=str,
                      keep_default_na=False, usecols=["entity_id"])["entity_id"]
    assert ids.is_unique
    alt = OUT / "alt"
    alt.mkdir(parents=True, exist_ok=True)
    for name, col in [("matching_results.tsv", "matched_entity_ids"),
                      ("candidate_pairs.tsv", "candidate_entity_ids")]:
        write_tsv(pd.DataFrame({"source1_entity_id": ids, col: ""}), alt / f"empty_{name}")
    print(f"wrote {len(ids):,} rows to output/alt/empty_matching_results.tsv and empty_candidate_pairs.tsv")


if __name__ == "__main__":
    main()
