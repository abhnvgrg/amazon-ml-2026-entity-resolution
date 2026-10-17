"""EXPERIMENT (D-010): recall of the blocking2.py views and their union, overall and per slice.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from blocking import countries
from blocking2 import RANK_NONE, VIEWS
from common import WORK
from translit import has_indic

KS = (1, 3, 5, 10, 20)


def gt_pairs(country: str) -> pd.DataFrame:
    gt = pd.read_parquet(WORK / "norm" / "train_gt.parquet")
    s1 = pd.read_parquet(WORK / "norm" / "train_S1.parquet", columns=["i", "name"],
                         filters=[("country", "==", country)]).set_index("i")["name"]
    gt = gt[gt["s1"].isin(s1.index)]
    recs = []
    for src in (2, 3):
        r = pd.read_parquet(WORK / "norm" / f"train_S{src}.parquet", columns=["i", "name", "addr_empty"],
                            filters=[("country", "==", country)])
        recs.append(r.assign(src=np.int8(src)))
    r = pd.concat(recs, ignore_index=True).rename(columns={"i": "rec"})
    g = gt.merge(r, on=["src", "rec"], how="left")
    s1n = s1.reindex(g["s1"]).tolist()
    g["indic"] = [has_indic(x) for x in g["name"].tolist()]
    g["name_sim_lt50"] = [fuzz.token_set_ratio(a, b) < 50 for a, b in zip(g["name"].tolist(), s1n)]
    return g.drop(columns="name")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    d = WORK / ("blocks2_bench" if a.bench else "blocks2")
    rows, rep = [], {}
    for c in countries("train"):
        b = pd.read_parquet(d / f"train_{c}{a.tag}.parquet")
        g = gt_pairs(c)
        if a.bench:
            q = b[["src", "rec"]].drop_duplicates()
            g = g.merge(q, on=["src", "rec"])
        hit = g.merge(b[["src", "rec", "s1", *[f"rank_{v}" for v in VIEWS]]], on=["src", "rec", "s1"], how="left")
        for v in VIEWS:
            hit[f"rank_{v}"] = hit[f"rank_{v}"].fillna(RANK_NONE)
        hit["union"] = hit[[f"rank_{v}" for v in VIEWS]].min(axis=1)
        slices = {"all": np.ones(len(hit), bool), "indic": hit["indic"].to_numpy(),
                  "addr_empty": hit["addr_empty"].to_numpy(), "name_sim<50": hit["name_sim_lt50"].to_numpy()}
        for sl, m in slices.items():
            h = hit[m]
            row = {"country": c, "slice": sl, "gt_pairs": int(m.sum())}
            for v in VIEWS:
                for k in KS:
                    row[f"{v}@{k}"] = float((h[f"rank_{v}"] < k).mean()) if len(h) else np.nan
            row["union"] = float((h["union"] < RANK_NONE).mean()) if len(h) else np.nan
            rows.append(row)
        n_rec = b[["src", "rec"]].drop_duplicates().shape[0]
        rep[c] = {"pairs": len(b), "records": n_rec, "cand_per_record": round(len(b) / n_rec, 2),
                  "per_view_found": {v: int((b[f"rank_{v}"] < RANK_NONE).sum()) for v in VIEWS}}
        del b, g, hit
    res = pd.DataFrame(rows)
    tot = []
    for sl, g in res.groupby("slice", sort=False):
        w = g["gt_pairs"].to_numpy()
        r = {"country": "ALL", "slice": sl, "gt_pairs": int(w.sum())}
        for col in res.columns[3:]:
            r[col] = float(np.nansum(g[col].to_numpy() * w) / w.sum())
        tot.append(r)
    res = pd.concat([pd.DataFrame(tot), res], ignore_index=True)
    pd.set_option("display.width", 250)
    cols = ["country", "slice", "gt_pairs", "n@1", "n@10", "a@1", "a@10", "s@10", "union"]
    print(res[cols].round(4).to_string(index=False))
    print(json.dumps(rep, indent=1))
    res.to_csv(WORK / f"eval_blocking2{'_bench' if a.bench else ''}{a.tag}.tsv", sep="\t",
               index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
