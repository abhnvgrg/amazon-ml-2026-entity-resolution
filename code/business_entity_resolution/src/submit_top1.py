"""sub-02: top-1 S1 per record plus a global cosine threshold -> output/*.tsv for the test split.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from blocking import countries
from common import OUT, WORK, MemTracker, write_tsv


def join_ids(s1_order: np.ndarray, p: pd.DataFrame) -> np.ndarray:
    ids = "S" + p["src"].astype(str) + "-" + p["rec"].astype(str)
    g = ids.groupby(p["s1"].to_numpy()).agg(",".join)
    return pd.Series(s1_order).map(g).fillna("").to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=float, required=True)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--tag", default="")
    ap.add_argument("--matching-only", action="store_true", help="keep candidate_pairs.tsv as is")
    a = ap.parse_args()
    mt = MemTracker()
    s1 = pd.read_parquet(WORK / "norm" / "test_S1.parquet", columns=["i"])["i"].to_numpy()
    assert len(np.unique(s1)) == len(s1)
    cand = []
    for c in countries("test"):
        b = pd.read_parquet(WORK / "blocks" / f"test_{c}_K{a.K}{a.tag}.parquet",
                            filters=[("rank", "==", 0)])
        cand.append(b.drop(columns="rank"))
        print(f"{c}: {len(b):,} candidates, {(b['cos'] >= a.t).sum():,} matches")
    cand = pd.concat(cand, ignore_index=True)
    assert not cand.duplicated(["src", "rec"]).any()
    mt.stage("load candidates")
    OUT.mkdir(exist_ok=True)
    s1_str = "S1-" + pd.Series(s1).astype(str)
    if not a.matching_only:
        write_tsv(pd.DataFrame({"source1_entity_id": s1_str,
                                "candidate_entity_ids": join_ids(s1, cand)}),
                  OUT / "candidate_pairs.tsv")
    m = cand[cand["cos"] >= a.t]
    write_tsv(pd.DataFrame({"source1_entity_id": s1_str,
                            "matched_entity_ids": join_ids(s1, m)}),
              OUT / "matching_results.tsv")
    n_nonempty = int((m["s1"].nunique()))
    size = (OUT / "matching_results.tsv").stat().st_size
    print(f"matching_results.tsv: {size:,} bytes ({size / 1e6:.2f} MB)")
    print(f"wrote {len(s1):,} rows; {len(m):,} matches over {n_nonempty:,} S1 "
          f"({1 - n_nonempty / len(s1):.3f} predicted singletons)")
    mt.stage("write outputs")


if __name__ == "__main__":
    main()
