"""sub-02b (D-009, superseded by D-012): smallest top-1 cosine threshold whose
matching_results.tsv fits a byte cap, with val and stress macro F0.5 at each threshold.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from blocking import countries
from common import SEED, WORK, is_val
from eval_blocking import rec_key
from metrics import macro_f05_counts

GRID = np.round(np.arange(0.60, 0.9001, 0.005), 3)
HEADER = len("source1_entity_id\tmatched_entity_ids\n")


def ndigits(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).astype(str).str.len().to_numpy()


def test_sizes(K: int) -> pd.DataFrame:
    s1 = pd.read_parquet(WORK / "norm" / "test_S1.parquet", columns=["i"])["i"].to_numpy()
    fixed = HEADER + int((3 + ndigits(s1) + 2).sum())
    c = pd.concat([pd.read_parquet(WORK / "blocks" / f"test_{c}_K{K}.parquet",
                                   columns=["s1", "rec", "cos"], filters=[("rank", "==", 0)])
                   for c in countries("test")], ignore_index=True)
    idlen = 3 + ndigits(c["rec"].to_numpy())
    cos, s1c = c["cos"].to_numpy(), c["s1"].to_numpy()
    rows = []
    for t in GRID:
        m = cos >= t
        n_rows = len(np.unique(s1c[m]))
        size = fixed + int(idlen[m].sum()) + int(m.sum()) - n_rows
        rows.append({"t": t, "test_matches": int(m.sum()), "test_nonempty": n_rows, "bytes": size})
    return pd.DataFrame(rows).set_index("t")


def val_scores(K: int) -> pd.DataFrame:
    s1 = np.sort(pd.read_parquet(WORK / "norm" / "train_S1.parquet", columns=["i"])["i"].to_numpy())
    val = s1[is_val(s1)]
    rng = np.random.default_rng(SEED)
    removed = np.sort(rng.choice(val, size=int(0.2 * len(val)), replace=False))
    kept = np.setdiff1d(val, removed)
    gt = pd.read_parquet(WORK / "norm" / "train_gt.parquet")
    gt_key = rec_key(gt["src"], gt["rec"])
    o = np.argsort(gt_key)
    gt_key, gt_s1 = gt_key[o], gt["s1"].to_numpy()[o]

    def ntrue(ents):
        return np.bincount(np.searchsorted(ents, gt_s1[np.isin(gt_s1, ents)]), minlength=len(ents))

    normal, stress = [], []
    for c in countries("train"):
        b = pd.read_parquet(WORK / "blocks" / f"train_{c}_K{K}.parquet")
        r0 = b[b["rank"] == 0]
        normal.append(r0[["src", "rec", "s1", "cos"]])
        bs = b[~np.isin(b["s1"].to_numpy(), removed)].sort_values(["src", "rec", "rank"])
        stress.append(bs.drop_duplicates(["src", "rec"])[["src", "rec", "s1", "cos"]])
        del b, bs
    out = {}
    for name, parts, ents in (("val", normal, val), ("stress", stress, kept)):
        p = pd.concat(parts, ignore_index=True)
        p = p[np.isin(p["s1"].to_numpy(), ents)]
        key = rec_key(p["src"], p["rec"])
        pos = np.searchsorted(gt_key, key).clip(max=len(gt_key) - 1)
        label = ((gt_key[pos] == key) & (gt_s1[pos] == p["s1"].to_numpy())).astype(np.float64)
        idx = np.searchsorted(ents, p["s1"].to_numpy())
        cos, nt = p["cos"].to_numpy(), ntrue(ents)
        rows = []
        for t in GRID:
            m = cos >= t
            r = macro_f05_counts(nt, np.bincount(idx[m], minlength=len(ents)),
                                 np.bincount(idx[m], weights=label[m], minlength=len(ents)))
            rows.append({"t": t, f"{name}_P": r["micro_precision"], f"{name}_R": r["micro_recall"],
                         f"{name}_F05": r["macro_f05"]})
        out[name] = pd.DataFrame(rows).set_index("t")
    return out["val"].join(out["stress"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-bytes", type=int, default=90_000_000)
    ap.add_argument("--K", type=int, default=10)
    a = ap.parse_args()
    res = test_sizes(a.K).join(val_scores(a.K))
    res["MB"] = (res["bytes"] / 1e6).round(2)
    pd.set_option("display.width", 200)
    print(res.drop(columns="bytes").round(4).to_string())
    ok = res[res["bytes"] <= a.max_bytes]
    t = float(ok.index.min())
    print(f"\nlowest t with size <= {a.max_bytes:,} bytes: t={t}")
    print(res.loc[t].to_string())
    res.to_csv(WORK / "size_threshold.tsv", sep="\t", lineterminator="\n")


if __name__ == "__main__":
    main()
