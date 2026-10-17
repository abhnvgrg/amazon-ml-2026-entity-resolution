"""Blocking v1 recall@K on train, plus the sub-02 top-1 / cosine-threshold matcher.

Recall@K is the share of ground-truth (record, S1) pairs whose S1 is in the record's top-K.
The sub-02 threshold is tuned on the train part of the S1-grouped split and reported on val.
Run: python src/eval_blocking.py --K 10
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from blocking import countries
from common import WORK, MemTracker, is_val
from metrics import macro_f05_counts

KS = (1, 3, 5, 10, 20, 50)
GRID = np.round(np.arange(0.20, 0.951, 0.025), 3)


def rec_key(src, rec) -> np.ndarray:
    rec = np.asarray(rec, np.int64)
    assert rec.max(initial=0) < 10**12
    return np.asarray(src, np.int64) * 10**12 + rec


def sweep(idx, cos, label, ntrue, mask) -> pd.DataFrame:
    rows = []
    for t in GRID:
        m = cos >= t
        npred = np.bincount(idx[m], minlength=len(ntrue))
        tp = np.bincount(idx[m], weights=label[m], minlength=len(ntrue))
        for name, mk in (("train", ~mask), ("val", mask)):
            r = macro_f05_counts(ntrue[mk], npred[mk], tp[mk])
            rows.append({"t": t, "part": name, **r})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    mt = MemTracker()
    s1 = pd.read_parquet(WORK / "norm" / "train_S1.parquet", columns=["i", "country"])
    s1 = s1.sort_values("i", ignore_index=True)
    s1_ids = s1["i"].to_numpy()
    gt = pd.read_parquet(WORK / "norm" / "train_gt.parquet")
    gt_key = rec_key(gt["src"], gt["rec"])
    o = np.argsort(gt_key)
    gt_key, gt_s1 = gt_key[o], gt["s1"].to_numpy()[o]
    gt_country = s1["country"].to_numpy()[np.searchsorted(s1_ids, gt_s1)]
    ntrue = np.bincount(np.searchsorted(s1_ids, gt_s1), minlength=len(s1_ids))
    del gt, o
    mt.stage("load gt")

    ks = [k for k in KS if k <= a.K]
    rep = {"K": a.K, "recall": {}, "pairs": {}}
    top1 = []
    for c in countries("train"):
        b = pd.read_parquet(WORK / "blocks" / f"train_{c}_K{a.K}{a.tag}.parquet")
        key = rec_key(b["src"], b["rec"])
        pos = np.searchsorted(gt_key, key).clip(max=len(gt_key) - 1)
        hit = (gt_key[pos] == key) & (gt_s1[pos] == b["s1"].to_numpy())
        rank = b["rank"].to_numpy()
        n_gt = int((gt_country == c).sum())
        for k in ks:
            rep["recall"][f"{c}@{k}"] = round(float(hit[rank < k].sum()) / max(n_gt, 1), 4)
            rep["pairs"][f"{c}@{k}"] = int((rank < k).sum())
        r0 = rank == 0
        top1.append(pd.DataFrame({"s1": b["s1"].to_numpy()[r0], "cos": b["cos"].to_numpy()[r0],
                                  "label": hit[r0]}))
        del b, key, pos, hit, rank
        mt.stage(f"recall {c}")
    for k in ks:
        tot_hit = sum(rep["recall"][f"{c}@{k}"] * int((gt_country == c).sum()) for c in countries("train"))
        rep["recall"][f"all@{k}"] = round(tot_hit / len(gt_key), 4)
        rep["pairs"][f"all@{k}"] = sum(rep["pairs"][f"{c}@{k}"] for c in countries("train"))
    print(json.dumps(rep, indent=1))

    t1 = pd.concat(top1, ignore_index=True)
    idx = np.searchsorted(s1_ids, t1["s1"].to_numpy())
    mask = is_val(s1_ids)
    res = sweep(idx, t1["cos"].to_numpy(), t1["label"].to_numpy().astype(np.float64), ntrue, mask)
    tr = res[res.part == "train"].set_index("t")
    val = res[res.part == "val"].set_index("t")
    best = float(tr["macro_f05"].idxmax())
    print(val.round(4).to_string())
    print(f"best t on train part = {best}: train F0.5 {tr.loc[best, 'macro_f05']:.4f} | "
          f"val P {val.loc[best, 'micro_precision']:.4f} R {val.loc[best, 'micro_recall']:.4f} "
          f"F0.5 {val.loc[best, 'macro_f05']:.4f}")
    cty = s1["country"].to_numpy()
    m = t1["cos"].to_numpy() >= best
    npred = np.bincount(idx[m], minlength=len(s1_ids))
    tp = np.bincount(idx[m], weights=t1["label"].to_numpy()[m].astype(np.float64), minlength=len(s1_ids))
    for c in countries("train"):
        mk = mask & (cty == c)
        print(f"  val {c}: {macro_f05_counts(ntrue[mk], npred[mk], tp[mk])}")
    rep.update(best_t=best, val=val.loc[best].to_dict())
    (WORK / f"eval_blocking_K{a.K}{a.tag}.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
