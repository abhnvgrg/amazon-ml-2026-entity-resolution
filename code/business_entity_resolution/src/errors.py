"""Diagnostic: val error breakdown at threshold t (feeds docs/ERRORS.md).

False negatives by cause (true S1 not in candidates / in candidates but prob < t / assigned to
another S1) and by slice (Indic name, empty address, DBA-style name, other); false positives
split into distractors and wrong S1; and the oracle macro F0.5 gain of fixing each cause.
Run: python src/errors.py --t 0.69
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from common import SEED, WORK
from features import rec_key
from matcher import FEATS, ntrue_for, val_entities
from metrics import macro_f05_counts


def rec_info(keys: np.ndarray, cols=("core", "addr", "dev", "addr_empty", "name")) -> pd.DataFrame:
    out = []
    for k in (2, 3):
        ids = np.unique(keys[keys // 10**12 == k] % 10**12)
        t = ds.dataset(WORK / "norm" / f"train_S{k}.parquet").to_table(
            columns=["i", *cols], filter=ds.field("i").isin(pa.array(ids, pa.int64())))
        d = t.to_pandas()
        d["key"] = rec_key(np.int8(k), d["i"].to_numpy())
        out.append(d.drop(columns="i"))
    return pd.concat(out, ignore_index=True).set_index("key")


def s1_info(ids: np.ndarray) -> pd.DataFrame:
    t = ds.dataset(WORK / "norm" / "train_S1.parquet").to_table(
        columns=["i", "name", "core", "addr"], filter=ds.field("i").isin(pa.array(np.unique(ids), pa.int64())))
    return t.to_pandas().set_index("i")


def score(ents, ntrue, s1, tp):
    idx = np.searchsorted(ents, s1)
    return macro_f05_counts(ntrue, np.bincount(idx, minlength=len(ents)),
                            np.bincount(idx, weights=tp.astype(float), minlength=len(ents)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=float, default=0.69)
    ap.add_argument("--n-ex", type=int, default=15)
    a = ap.parse_args()
    val_ids, val_cty, _, gt_s1 = val_entities()
    gt = pd.read_parquet(WORK / "norm" / "train_gt.parquet")
    gt = gt[np.isin(gt["s1"].to_numpy(), val_ids)].reset_index(drop=True)
    gt["key"] = rec_key(gt["src"].to_numpy().astype(np.int8), gt["rec"].to_numpy())
    gt["country"] = val_cty[np.searchsorted(val_ids, gt["s1"].to_numpy())]
    best = pd.read_parquet(WORK / "best_val.parquet")
    incand = np.unique(np.concatenate([
        pd.read_parquet(FEATS / f"val_{c}.parquet", columns=["key", "label"]).query("label")["key"].to_numpy()
        for c in np.unique(val_cty)]))
    b = best.set_index("key")
    bp = b["prob"].reindex(gt["key"]).to_numpy()
    bs = b["s1"].reindex(gt["key"]).to_numpy()
    pred = ~np.isnan(bp) & (bp >= a.t)
    hit = pred & (bs == gt["s1"].to_numpy())
    gt["cause"] = np.select([hit, ~np.isin(gt["key"].to_numpy(), incand), pred],
                            ["TP", "(1) not in candidates", "(3) assigned to other S1"],
                            "(2) in candidates, prob < t")

    ri = rec_info(gt["key"].to_numpy()).reindex(gt["key"])
    si = s1_info(gt["s1"].to_numpy()).reindex(gt["s1"])
    nsim = cpdist(ri["core"].fillna("").tolist(), si["core"].fillna("").tolist(),
                  scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
    gt["slice"] = np.select([ri["dev"].fillna(False).to_numpy().astype(bool),
                             ri["addr_empty"].fillna(False).to_numpy().astype(bool), nsim < 50],
                            ["Devanagari name", "empty address", "name_sim<50 (DBA)"], "other")

    pd.set_option("display.width", 220)
    fn = gt[gt["cause"] != "TP"]
    print(f"t={a.t}: val S1 {len(val_ids):,}, GT pairs {len(gt):,}, TP {int((gt['cause'] == 'TP').sum()):,}, "
          f"FN {len(fn):,}")
    print("\nFN by cause x country (count, share of all GT pairs of that country):")
    ct = pd.crosstab(fn["cause"], fn["country"], margins=True)
    print(ct.to_string())
    print((ct / gt["country"].value_counts().reindex(ct.columns, fill_value=len(gt))).round(4).to_string())
    print("\nFN by cause x slice (count):")
    print(pd.crosstab([fn["country"], fn["cause"]], fn["slice"], margins=True).to_string())
    print("\nGT pairs per slice and recall by slice:")
    rs = gt.groupby(["country", "slice"]).agg(pairs=("cause", "size"),
                                              recall=("cause", lambda x: (x == "TP").mean()))
    print(rs.round(4).to_string())

    p = best[(best["prob"] >= a.t) & np.isin(best["s1"].to_numpy(), val_ids)].reset_index(drop=True)
    fp = p[~p["label"]].copy()
    fp["kind"] = np.where(fp["true_s1"] < 0, "distractor", "wrong S1")
    print(f"\npredictions to val S1: {len(p):,}, FP {len(fp):,} ({len(fp) / len(p):.4f})")
    print(pd.crosstab(fp["kind"], fp["country"], margins=True).to_string())

    ents_all = val_ids
    base_s1, base_tp = p["s1"].to_numpy(), p["label"].to_numpy()
    fixes = {}
    for cause in ["(1) not in candidates", "(2) in candidates, prob < t", "(3) assigned to other S1"]:
        add = gt[gt["cause"] == cause]
        s1 = np.r_[base_s1, add["s1"].to_numpy()]
        tp = np.r_[base_tp, np.ones(len(add), bool)]
        if cause.startswith("(3)"):
            drop = np.isin(p["key"].to_numpy(), add["key"].to_numpy()) & ~base_tp
            s1, tp = np.r_[base_s1[~drop], add["s1"].to_numpy()], np.r_[base_tp[~drop], np.ones(len(add), bool)]
        fixes[f"FN {cause}"] = (s1, tp)
    for kind in ["distractor", "wrong S1"]:
        keep = ~(~base_tp & np.isin(p["key"].to_numpy(), fp.loc[fp["kind"] == kind, "key"].to_numpy()))
        fixes[f"FP {kind} removed"] = (base_s1[keep], base_tp[keep])
    rows = []
    for c in [*np.unique(val_cty), "all"]:
        ents = ents_all if c == "all" else ents_all[val_cty == c]
        nt = ntrue_for(ents, gt_s1)
        sel = lambda s: np.isin(s, ents)
        m = sel(base_s1)
        r0 = score(ents, nt, base_s1[m], base_tp[m])["macro_f05"]
        row = {"country": c, "baseline": r0}
        for name, (s1, tp) in fixes.items():
            m = sel(s1)
            row[name] = score(ents, nt, s1[m], tp[m])["macro_f05"] - r0
        rows.append(row)
    print("\nmacro F0.5 baseline and oracle gain per fix (each fix alone):")
    print(pd.DataFrame(rows).set_index("country").round(4).T.to_string())

    rng = np.random.default_rng(SEED)
    ex = pd.concat([g.iloc[rng.choice(len(g), min(a.n_ex, len(g)), replace=False)]
                    for _, g in fp.groupby("kind")])
    rinf = rec_info(ex["key"].to_numpy(), cols=("name", "addr")).reindex(ex["key"])
    sp = s1_info(ex["s1"].to_numpy()).reindex(ex["s1"])
    st = s1_info(ex["true_s1"].to_numpy()[ex["true_s1"].to_numpy() >= 0]).reindex(ex["true_s1"])
    print(f"\nFP examples ({a.n_ex} per kind): record | predicted S1 | true S1")
    for j, (_, e) in enumerate(ex.iterrows()):
        print(f"[{e['kind']}] {e['country']} p={e['prob']:.3f}\n"
              f"   rec : {rinf['name'].iat[j]} | {rinf['addr'].iat[j]}\n"
              f"   pred: {sp['name'].iat[j]} | {sp['addr'].iat[j]}"
              + (f"\n   true: {st['name'].iat[j]} | {st['addr'].iat[j]}" if e["true_s1"] >= 0 else ""))


if __name__ == "__main__":
    main()
