"""EXPERIMENT (sub-07, abandoned): export normalized record texts for the e5 encoder.

Runs in the main environment and writes one text per line plus a key array per split,
country and side to work/embeddings/, for embed.py to encode.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from blocking import countries
from common import WORK, MemTracker
from features import rec_key

EMB = WORK / "embeddings"


def text(df: pd.DataFrame) -> list[str]:
    out = (df["name"].str.strip() + " " + df["addr"].str.strip()).str.strip().str.lower()
    out = out.str.replace(r"\s+", " ", regex=True)
    return out.tolist()


def write(split: str, c: str, kind: str, keys: np.ndarray, texts: list[str]) -> None:
    assert len(keys) == len(texts) and not any("\n" in t for t in texts)
    np.save(EMB / f"{split}_{c}_{kind}_keys.npy", keys)
    (EMB / f"{split}_{c}_{kind}.txt").write_text("\n".join(texts) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["train", "test"])
    a = ap.parse_args()
    mt = MemTracker()
    EMB.mkdir(parents=True, exist_ok=True)
    for split in a.splits:
        for c in countries(split):
            s1 = pd.read_parquet(WORK / "norm" / f"{split}_S1.parquet", columns=["i", "name", "addr"],
                                 filters=[("country", "==", c)]).sort_values("i", ignore_index=True)
            write(split, c, "S1", s1["i"].to_numpy(), text(s1))
            parts = []
            for k in (2, 3):
                d = pd.read_parquet(WORK / "norm" / f"{split}_S{k}.parquet", columns=["i", "name", "addr"],
                                    filters=[("country", "==", c)])
                d["key"] = rec_key(np.int8(k), d.pop("i").to_numpy())
                parts.append(d)
            r = pd.concat(parts, ignore_index=True).sort_values("key", ignore_index=True)
            write(split, c, "R", r["key"].to_numpy(), text(r))
            mt.stage(f"{split} {c}: S1 {len(s1):,}, records {len(r):,}")
            del s1, r, parts


if __name__ == "__main__":
    main()
