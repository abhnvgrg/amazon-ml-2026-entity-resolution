"""Stream the raw TSVs into normalized parquet caches (work/norm/{split}_{src}.parquet).

Columns: int64 id (prefix stripped), country, name, core, addr, Indic flag, empty-address
flag. The train split also writes work/norm/train_gt.parquet, one row per matched record.
Run: python src/prep.py --split train|test
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common import DATA, RD, SRCS, WORK, MemTracker
from normalize import normalize_df

DEV_RE = re.compile(r"[ऀ-ॿ]")


def prep(split: str, chunk: int = 500_000) -> None:
    mt = MemTracker()
    (WORK / "norm").mkdir(parents=True, exist_ok=True)
    for src in SRCS:
        out = WORK / "norm" / f"{split}_{src}.parquet"
        writer = None
        n = 0
        for ch in pd.read_csv(DATA / split / f"{split}_source{src[1]}.tsv", chunksize=chunk, **RD):
            ids = ch["entity_id"].str.slice(3)
            assert ch["entity_id"].str.startswith(src + "-").all()
            i = ids.astype(np.int64)
            assert (i.astype(str) == ids).all(), "non-canonical numeric ids"
            ch = normalize_df(ch)
            t = pd.DataFrame({
                "i": i.to_numpy(), "country": ch["country"].to_numpy(),
                "name": ch["name"], "core": ch["core"], "addr": ch["addr"],
                "dev": ch["business_name"].str.contains(DEV_RE).to_numpy(),
                "addr_empty": (ch["business_address"].str.strip() == "").to_numpy(),
            })
            tb = pa.Table.from_pandas(t, preserve_index=False)
            writer = writer or pq.ParquetWriter(out, tb.schema)
            writer.write_table(tb)
            n += len(t)
        writer.close()
        mt.stage(f"prep {split} {src} ({n:,} rows)")


def prep_gt() -> None:
    s1s, srcs, recs = [], [], []
    with open(DATA / "train" / "train_ground_truth.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            a, m = line.rstrip("\n").split("\t")
            s1 = int(a[3:])
            for x in m.split(","):
                if x:
                    s1s.append(s1)
                    srcs.append(int(x[1]))
                    recs.append(int(x[3:]))
    gt = pd.DataFrame({"src": np.array(srcs, np.int8), "rec": np.array(recs, np.int64),
                       "s1": np.array(s1s, np.int64)})
    assert not gt.duplicated(["src", "rec"]).any()
    gt.to_parquet(WORK / "norm" / "train_gt.parquet", index=False)
    print(f"gt: {len(gt):,} pairs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--gt-only", action="store_true")
    a = ap.parse_args()
    if not a.gt_only:
        prep(a.split)
    if a.split == "train":
        prep_gt()
