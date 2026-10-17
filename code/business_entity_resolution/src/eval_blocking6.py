"""Sweep of the extra name-only (rank_n, sub-06) or dense (rank_d, sub-07) candidate depth.

Reports overall and empty-address pair recall, candidates per S1 and extra pairs per record
on the val-sample S1, on top of the chosen blocking v2 rule.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import WORK
from eval_blocking3 import rec_rank
from features import V3_DELTA, V3_K, V3_KS, rec_key


def empty_keys(c: str) -> np.ndarray:
    out = []
    for k in (2, 3):
        t = pd.read_parquet(WORK / "norm" / f"train_S{k}.parquet", columns=["i", "addr"], filters=[("country", "==", c)])
        out.append(rec_key(np.int8(k), t.loc[t["addr"].str.len() == 0, "i"].to_numpy()))
    return np.sort(np.concatenate(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="*", default=["India", "US"])
    ap.add_argument("--kn", type=int, nargs="*", default=[0, 1, 3, 5])
    ap.add_argument("--col", default="rank_n", help="extra source rank column to sweep (rank_n or rank_d)")
    a = ap.parse_args()
    gt_all = pd.read_parquet(WORK / "norm" / "train_gt.parquet")
    gt_all["key"] = rec_key(gt_all["src"].to_numpy().astype(np.int8), gt_all["rec"].to_numpy())
    rows = []
    for c in a.countries:
        u = pd.read_parquet(WORK / "blocks3" / f"train_{c}.parquet", columns=["src", "rec", "s1", "cos", "rank_s", a.col])
        key = rec_key(u["src"].to_numpy(), u["rec"].to_numpy())
        rr = rec_rank(key)
        start = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
        grp = np.repeat(np.arange(len(start)), np.diff(np.r_[start, len(key)]))
        cos = u["cos"].to_numpy()
        gap = cos[start][grp] - cos
        rs, rn, s1 = u["rank_s"].to_numpy(), u[a.col].to_numpy(), u["s1"].to_numpy()
        del u, grp, start
        base = ((rr < V3_K) & ((rr == 0) | (gap <= V3_DELTA))) | (rs < V3_KS)
        ek = empty_keys(c)
        is_e = np.isin(key, ek)
        n_e = len(ek)
        n_r = len(np.unique(key))
        ents = pd.read_parquet(WORK / "feats" / f"val_sample_{c}.parquet")["s1"].to_numpy()
        gt = gt_all[np.isin(gt_all["s1"].to_numpy(), ents)]
        ge = np.isin(gt["key"].to_numpy(), ek)
        sel = np.isin(s1, ents)
        gidx = pd.MultiIndex.from_arrays([gt["key"].to_numpy(), gt["s1"].to_numpy()])
        uidx = pd.MultiIndex.from_arrays([key[sel], s1[sel]])
        for kn in a.kn:
            keep = base | (rn < kn)
            kept = pd.MultiIndex.from_arrays([key[sel][keep[sel]], s1[sel][keep[sel]]])
            hit = gidx.isin(kept)
            rows.append({"country": c, "K_n": kn, "recall": hit.mean(), "recall_empty_addr": hit[ge].mean(),
                         "gt_pairs": len(gt), "empty_gt_pairs": int(ge.sum()), "cand_per_s1": keep[sel].sum() / len(ents),
                         "extra_per_empty_rec": (keep & ~base & is_e).sum() / n_e,
                         "extra_per_rec": (keep & ~base).sum() / n_r,
                         "union_recall_empty": gidx[ge].isin(uidx).mean()})
        print(f"{c}: {n_e:,} empty-address records; {a.col} pairs in union {int((rn < 127).sum()):,}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(WORK / f"eval_blocking6_{a.col}.tsv", sep="\t", index=False, lineterminator="\n")
    tot = res.groupby("K_n").apply(lambda g: pd.Series({
        "recall": np.average(g["recall"], weights=g["gt_pairs"]),
        "recall_empty_addr": np.average(g["recall_empty_addr"], weights=g["empty_gt_pairs"]),
        "cand_per_s1": g["cand_per_s1"].mean(), "extra_per_empty_rec": g["extra_per_empty_rec"].mean(),
        "extra_per_rec": g["extra_per_rec"].mean()}))
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))
    print(tot.round(4).to_string())


if __name__ == "__main__":
    main()
