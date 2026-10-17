"""Sweep of the blocking v2 pruning rule on train (D-015).

Keep a pair if (record rank < K and (rank 0 or top1 cos - cos <= delta)) or S1-side rank < ks
or address rank < ka. Reports pair recall and kept candidates per S1 for each setting; the
chosen rule is delta 0.1, K 10, ks 5, ka 0.
Run: python src/eval_blocking3.py --val-sample
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from common import WORK
from features import rec_key


def rec_rank(key: np.ndarray) -> np.ndarray:
    start = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
    cnt = np.diff(np.r_[start, len(key)])
    return np.arange(len(key)) - np.repeat(start, cnt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="*", default=["India", "US"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--val-sample", action="store_true", help="evaluate on the matcher's val-sample S1 only")
    ap.add_argument("--deltas", type=float, nargs="*", default=[0.05, 0.1, 0.15, 0.2, 0.3, 1.0])
    ap.add_argument("--kmax", type=int, nargs="*", default=[3, 5, 10])
    ap.add_argument("--ks", type=int, nargs="*", default=[0, 3, 5, 10])
    ap.add_argument("--ka", type=int, nargs="*", default=[0, 1, 3])
    a = ap.parse_args()
    gt_all = pd.read_parquet(WORK / "norm" / "train_gt.parquet")
    gt_all["key"] = rec_key(gt_all["src"].to_numpy().astype(np.int8), gt_all["rec"].to_numpy())
    rows = []
    for c in a.countries:
        u = pd.read_parquet(WORK / "blocks3" / f"train_{c}{a.tag}.parquet")
        u["key"] = rec_key(u["src"].to_numpy(), u["rec"].to_numpy())
        ents = np.unique(u.loc[u["rank_s"] < 127, "s1"].to_numpy())
        if a.val_sample:
            ents = np.intersect1d(ents, pd.read_parquet(WORK / "feats" / f"val_sample_{c}.parquet")["s1"].to_numpy())
        gt = gt_all[np.isin(gt_all["s1"].to_numpy(), ents)]
        u = u[np.isin(u["s1"].to_numpy(), ents)].reset_index(drop=True)
        gpid = pd.MultiIndex.from_arrays([gt["key"].to_numpy(), gt["s1"].to_numpy()])
        upid = pd.MultiIndex.from_arrays([u["key"].to_numpy(), u["s1"].to_numpy()])
        is_true = upid.isin(gpid)
        n_s1, n_gt = len(ents), len(gt)
        old = pd.read_parquet(WORK / "blocks" / f"train_{c}_K10.parquet", columns=["src", "rec", "s1", "rank", "cos"])
        old = old[np.isin(old["s1"].to_numpy(), ents)]
        old = old[(old["rank"] < 5) & ((old["rank"] == 0) | (old["cos"] >= 0.3))]
        opid = pd.MultiIndex.from_arrays([rec_key(old["src"].to_numpy(), old["rec"].to_numpy()), old["s1"].to_numpy()])
        rows.append({"country": c, "rule": "sub-03 (v1 pruned)", "recall": gpid.isin(opid).mean(),
                     "cand_per_s1": len(old) / n_s1, "s1": n_s1, "gt_pairs": n_gt})
        del old
        full = pd.read_parquet(WORK / "blocks3" / f"train_{c}{a.tag}.parquet", columns=["src", "rec", "s1", "cos"])
        full["key"] = rec_key(full["src"].to_numpy(), full["rec"].to_numpy())
        full = full[np.isin(full["key"].to_numpy(), np.unique(u["key"].to_numpy()))]
        rr = rec_rank(full["key"].to_numpy())
        top1 = pd.Series(full["cos"].to_numpy()[rr == 0], index=full["key"].to_numpy()[rr == 0])
        rk = pd.Series(rr, index=pd.MultiIndex.from_arrays([full["key"].to_numpy(), full["s1"].to_numpy()]))
        u["rrank"] = rk.reindex(upid).to_numpy()
        u["gap"] = top1.reindex(u["key"]).to_numpy() - u["cos"].to_numpy()
        del full, rk, top1
        rows.append({"country": c, "rule": "union (no pruning)", "recall": is_true.sum() / n_gt,
                     "cand_per_s1": len(u) / n_s1, "s1": n_s1, "gt_pairs": n_gt})
        rrank, gap, rs, ra = (u[x].to_numpy() for x in ("rrank", "gap", "rank_s", "rank_a"))
        for d, k, ks, ka in itertools.product(a.deltas, a.kmax, a.ks, a.ka):
            keep = ((rrank < k) & ((rrank == 0) | (gap <= d))) | (rs < ks) | (ra < ka)
            rows.append({"country": c, "rule": f"delta={d} K={k} s<{ks} a<{ka}",
                         "recall": is_true[keep].sum() / n_gt, "cand_per_s1": keep.sum() / n_s1,
                         "s1": n_s1, "gt_pairs": n_gt})
    res = pd.DataFrame(rows)
    res.to_csv(WORK / f"eval_blocking3{a.tag}.tsv", sep="\t", index=False, lineterminator="\n")
    pd.set_option("display.width", 200)
    for c, g in res.groupby("country"):
        print(f"\n== {c}: {g['s1'].iat[0]:,} S1, {g['gt_pairs'].iat[0]:,} GT pairs")
        print(g.head(2)[["rule", "recall", "cand_per_s1"]].round(4).to_string(index=False))
        base = g.iloc[0]
        sw = g.iloc[2:]
        print("best recall at <= sub-03 candidates/S1:")
        print(sw[sw["cand_per_s1"] <= base["cand_per_s1"]].nlargest(8, "recall")[["rule", "recall", "cand_per_s1"]]
              .round(4).to_string(index=False))
        print("fewest candidates/S1 at >= sub-03 recall + 0.01:")
        print(sw[sw["recall"] >= base["recall"] + 0.01].nsmallest(8, "cand_per_s1")[["rule", "recall", "cand_per_s1"]]
              .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
