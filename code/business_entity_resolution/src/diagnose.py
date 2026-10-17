"""Diagnostic: sub-02 error analysis on val and a leakage check on ID / row order.

Splits false positives into distractors (the record has no true S1) and wrong-S1 assignments,
and prints distractor pattern counts. The leakage check measures whether IDs or file row
positions of matched records correlate with their S1's (they don't; never used as features).
Run: python src/diagnose.py --t 0.675 [--examples]
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from blocking import countries
from common import SEED, WORK, is_val
from eval_blocking import rec_key

Q = [0.05, 0.25, 0.5, 0.75, 0.95]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=float, default=0.675)
    ap.add_argument("--K", type=int, default=10)
    a = ap.parse_args()
    gt = pd.read_parquet(WORK / "norm" / "train_gt.parquet")
    gt_key = rec_key(gt["src"], gt["rec"])
    o = np.argsort(gt_key)
    gt_key, gt_s1 = gt_key[o], gt["s1"].to_numpy()[o]

    parts = []
    for c in countries("train"):
        b = pd.read_parquet(WORK / "blocks" / f"train_{c}_K{a.K}.parquet")
        key = rec_key(b["src"], b["rec"])
        pos = np.searchsorted(gt_key, key).clip(max=len(gt_key) - 1)
        has = gt_key[pos] == key
        true_s1 = np.where(has, gt_s1[pos], -1)
        b["true_s1"] = true_s1
        b["hit"] = b["s1"].to_numpy() == true_s1
        b["true_in_k"] = b.groupby(["src", "rec"])["hit"].transform("any")
        r0 = b[b["rank"] == 0].copy()
        r0["country"] = c
        r1 = b[b["rank"] == 1].set_index(["src", "rec"])["cos"]
        r0["gap"] = r0["cos"].to_numpy() - r1.reindex(pd.MultiIndex.from_frame(r0[["src", "rec"]])).fillna(0).to_numpy()
        parts.append(r0)
        del b
    r0 = pd.concat(parts, ignore_index=True)
    pv = r0[is_val(r0["s1"].to_numpy())]
    pred = pv[pv["cos"] >= a.t]
    kind = np.select(
        [pred["hit"], pred["true_s1"] < 0, pred["true_in_k"]],
        ["TP", "FP(a) distractor", "FP(b1) wrong S1, true in top-K"], "FP(b2) wrong S1, true not retrieved")
    pred = pred.assign(kind=kind)
    lines = [f"sub-02 val predictions at t={a.t}: {len(pred):,}"]
    tab = pred.groupby(["kind"]).size().rename("n").to_frame()
    tab["share"] = (tab["n"] / len(pred)).round(4)
    lines.append(tab.to_string())
    lines.append("\nper country FP share of predictions:")
    lines.append(pd.crosstab(pred["country"], pred["kind"], normalize="index").round(4).to_string())
    lines.append("\ncos quantiles " + str(Q))
    for k, g in pred.groupby("kind"):
        lines.append(f"  {k:40s} cos {np.round(g['cos'].quantile(Q).to_numpy(), 3)}  gap {np.round(g['gap'].quantile(Q).to_numpy(), 3)}")
    lines.append(f"\nall val rank-0 records: distractor share {float((pv['true_s1'] < 0).mean()):.4f}")

    lines.append("\nLeakage check (diagnostic only):")
    s1 = pd.read_parquet(WORK / "norm" / "train_S1.parquet", columns=["i"])["i"].to_numpy()
    s1_pos = pd.Series(np.arange(len(s1)), index=s1)
    for src in (2, 3):
        rec = pd.read_parquet(WORK / "norm" / f"train_S{src}.parquet", columns=["i"])["i"].to_numpy()
        rec_pos = pd.Series(np.arange(len(rec)), index=rec)
        g = gt[gt["src"] == src].sample(200_000, random_state=42)
        rp = rec_pos.reindex(g["rec"]).to_numpy() / len(rec)
        sp_ = s1_pos.reindex(g["s1"]).to_numpy() / len(s1)
        lines.append(f"  S{src}: corr(row pos rec, row pos S1) = {np.corrcoef(rp, sp_)[0, 1]:+.4f}; "
                     f"corr(ID rec, ID S1) = {np.corrcoef(g['rec'], g['s1'])[0, 1]:+.4f}; "
                     f"median |d_rowpos| = {np.median(np.abs(rp - sp_)):.3f} (random ~0.293)")
        matched = np.isin(rec, gt.loc[gt["src"] == src, "rec"].to_numpy())
        dec = np.array_split(matched, 10)
        lines.append(f"      match rate by file decile: {[round(float(d.mean()), 3) for d in dec]}")
        idq = pd.qcut(rec, 10, labels=False)
        lines.append(f"      match rate by ID decile:   {[round(float(matched[idq == k].mean()), 3) for k in range(10)]}")
        m = pd.Series(rec).map(gt[gt["src"] == src].set_index("rec")["s1"])
        same_next = (m.to_numpy()[1:] == m.to_numpy()[:-1]) & m.notna().to_numpy()[1:]
        lines.append(f"      P(next row has same S1) = {same_next.mean():.5f}")
    txt = "\n".join(lines)
    print(txt)
    (WORK / "diagnose_sub02.txt").write_text(txt, encoding="utf-8")


NUM_RE = re.compile(r"\d+")


def _pattern(rn, ra, sn, sa) -> str:
    ns = fuzz.token_set_ratio(rn, sn)
    as_ = fuzz.token_set_ratio(ra, sa)
    same_nums = set(NUM_RE.findall(ra)) == set(NUM_RE.findall(sa)) and ra != ""
    if ns >= 80 and not same_nums:
        return "similar name, different address numbers"
    if ns >= 80 and same_nums:
        return "similar name, same address numbers"
    if ns < 80 and as_ >= 80:
        return "different name, similar address"
    return "other"


def examples(t: float, K: int = 10, n_class: int = 100_000) -> None:
    gt = pd.read_parquet(WORK / "norm" / "train_gt.parquet")
    gt_key = np.sort(rec_key(gt["src"], gt["rec"]))
    parts = []
    for c in countries("train"):
        b = pd.read_parquet(WORK / "blocks" / f"train_{c}_K{K}.parquet", filters=[("rank", "==", 0)])
        b = b[is_val(b["s1"].to_numpy()) & (b["cos"] >= t)]
        key = rec_key(b["src"], b["rec"])
        pos = np.searchsorted(gt_key, key).clip(max=len(gt_key) - 1)
        b = b.assign(distr=gt_key[pos] != key, country=c)
        parts.append(b)
    p = pd.concat(parts, ignore_index=True)
    gts = gt.set_index(["src", "rec"])["s1"]
    true_s1 = gts.reindex(pd.MultiIndex.from_frame(p[["src", "rec"]])).to_numpy()
    p["tp"] = true_s1 == p["s1"].to_numpy()
    d = p[p["distr"]].sample(min(n_class, int(p["distr"].sum())), random_state=SEED)
    tp = p[p["tp"] & p["cos"].between(0.75, 0.85)].sample(30, random_state=SEED)
    need = pd.concat([d, tp])
    s1 = pd.read_parquet(WORK / "norm" / "train_S1.parquet", columns=["i", "core", "addr"],
                         filters=[("i", "in", need["s1"].unique().tolist())]).set_index("i")
    rec = pd.concat([pd.read_parquet(WORK / "norm" / f"train_S{k}.parquet", columns=["i", "core", "addr"],
                                     filters=[("i", "in", need.loc[need.src == k, "rec"].tolist())]).assign(src=k)
                     for k in (2, 3)]).set_index(["src", "i"])

    def texts(df):
        r = rec.loc[list(zip(df["src"], df["rec"]))]
        s = s1.loc[df["s1"]]
        return r["core"].tolist(), r["addr"].tolist(), s["core"].tolist(), s["addr"].tolist()

    rn, ra, sn, sa = texts(d)
    d = d.assign(pattern=[_pattern(*x) for x in zip(rn, ra, sn, sa)], rn=rn, ra=ra, sn=sn, sa=sa)
    lines = [f"Distractor FPs at t={t} on val: {int(p['distr'].sum()):,} (pattern counts on a {len(d):,} sample)"]
    tab = pd.crosstab(d["pattern"], d["country"], margins=True, normalize="columns").round(3)
    lines.append(tab.to_string())
    lines.append("\nmedian cos by pattern: " + d.groupby("pattern")["cos"].median().round(3).to_string())
    lines.append("\n30 distractor examples (cos | record core | record addr || S1 core | S1 addr | pattern):")
    for _, x in d.sample(30, random_state=1).iterrows():
        lines.append(f"  {x.cos:.3f} | {x.rn} | {x.ra} || {x.sn} | {x.sa} | {x.pattern}")
    rn, ra, sn, sa = texts(tp)
    lines.append("\n30 true matches with cos 0.75-0.85:")
    for x, a1, a2, b1, b2 in zip(tp.itertuples(), rn, ra, sn, sa):
        lines.append(f"  {x.cos:.3f} | {a1} | {a2} || {b1} | {b2} | {_pattern(a1, a2, b1, b2)}")
    txt = "\n".join(lines)
    print(txt)
    (WORK / "diagnose_examples.txt").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    import sys
    if "--examples" in sys.argv:
        examples(0.675)
    else:
        main()
