"""Learn an Indic-token -> Latin-token dictionary and map Indic-script record names (D-014).

The dictionary is learned from TRAIN-split ground-truth pairs only (never validation): an
Indic-script record name and a Latin S1 name with equal token counts, aligned in order; maps
with count >= 3 and share >= 0.7 are kept (work/indic_dict.json). The name core of every
Indic-script S2/S3 record is then rewritten with it (unmapped tokens fall back to
translit.py); the original is kept in `core_raw`, so the step is idempotent.
Run after prep.py, before blocking: python src/indic_dict.py
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import WORK, MemTracker, is_val
from normalize import LEGAL
from translit import has_indic, translit

DICT = WORK / "indic_dict.json"
MIN_N, MIN_SHARE = 3, 0.7


def learn() -> dict[str, str]:
    s1 = pd.read_parquet(WORK / "norm" / "train_S1.parquet", columns=["i", "name"]).set_index("i")["name"]
    gt = pd.read_parquet(WORK / "norm" / "train_gt.parquet")
    gt = gt[~is_val(gt["s1"].to_numpy())]
    cnt: dict[str, Counter] = defaultdict(Counter)
    n_used = 0
    for k in (2, 3):
        r = pd.read_parquet(WORK / "norm" / f"train_S{k}.parquet", columns=["i", "name"]).set_index("i")["name"]
        g = gt[gt["src"].to_numpy() == k]
        for a, b in zip(r.reindex(g["rec"]).to_numpy(), s1.reindex(g["s1"]).to_numpy()):
            if not isinstance(a, str) or not isinstance(b, str) or not has_indic(a) or has_indic(b):
                continue
            ta, tb = a.split(), b.split()
            if len(ta) != len(tb):
                continue
            n_used += 1
            for x, y in zip(ta, tb):
                if has_indic(x):
                    cnt[x][y] += 1
    d = {}
    for x, c in cnt.items():
        n = sum(c.values())
        y, m = c.most_common(1)[0]
        if n >= MIN_N and m / n >= MIN_SHARE:
            d[x] = y
    print(f"dictionary: {n_used:,} aligned train-split pairs, {len(cnt):,} Indic token types, {len(d):,} kept")
    DICT.write_text(json.dumps(d, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return d


def map_name(s: str, d: dict) -> str:
    out = [m for m in (d.get(t) or translit(t) for t in s.split()) if m not in LEGAL]
    return " ".join(out) or translit(s)


def apply(split: str, d: dict, mt: MemTracker) -> None:
    for k in (2, 3):
        path = WORK / "norm" / f"{split}_S{k}.parquet"
        df = pq.read_table(path).to_pandas()
        if "core_raw" not in df.columns:
            df["core_raw"] = df["core"]
        ind = np.fromiter((has_indic(x) for x in df["name"]), bool, len(df))
        df["core"] = df["core_raw"]
        df.loc[ind, "core"] = [map_name(x, d) for x in df.loc[ind, "name"]]
        tmp = path.with_suffix(".tmp.parquet")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        mt.stage(f"{split} S{k}: {int(ind.sum()):,} of {len(df):,} Indic names mapped")
        del df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["train", "test"])
    a = ap.parse_args()
    mt = MemTracker()
    d = learn()
    for s in a.splits:
        apply(s, d, mt)


if __name__ == "__main__":
    main()
