"""EXPERIMENT (sub-05, D-018, not adopted): group-level joint assignment.

Accepts or rejects whole predicted clusters by size-dependent mean-probability thresholds and
re-routes rejected members to their second-best S1. Tuned on half of the val S1; it gave no
gain over per-record argmax on the other half.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from blocking3 import mix64
from common import OUT, WORK
from errors import rec_info, s1_info
from matcher import join_ids, ntrue_for, val_entities, write_tsv
from metrics import macro_f05_counts
from translit import has_indic

BUCKETS = np.array([1, 2, 3, 5, 7])
T_GRID = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97]
FLOOR_GRID = [0.3, 0.5, 0.6, 0.71]
SOLO_GRID = [0.71, 0.8, 0.9, 0.95, 0.97, 0.99, 1.01]
PARAMS = WORK / "group_params.json"



def bucket(n: np.ndarray) -> np.ndarray:
    return np.searchsorted(BUCKETS, n, side="right") - 1


def assign(b: pd.DataFrame, p: dict) -> np.ndarray:
    T = np.asarray(p["T"])
    s1, pr, s2, pr2 = (b[c].to_numpy() for c in ("s1", "prob", "s1_2", "prob_2"))
    mem = pr >= p["floor"]
    g = pd.DataFrame({"s1": s1[mem], "prob": pr[mem]}).groupby("s1")["prob"].agg(["size", "sum"])
    n = g["size"].reindex(s1).fillna(0).to_numpy()
    tot = g["sum"].reindex(s1).fillna(0).to_numpy()
    ok_group = mem & (tot >= T[bucket(np.maximum(n, 1))] * n)
    out = np.where(ok_group | (mem & (pr >= p["solo"])), s1, -1)
    rest = mem & (out < 0) & (pr2 >= p["floor"]) & (s2 >= 0)
    if rest.any():
        n2 = g["size"].reindex(s2[rest]).fillna(0).to_numpy() + 1
        t2 = g["sum"].reindex(s2[rest]).fillna(0).to_numpy() + pr2[rest]
        ok2 = t2 >= T[bucket(n2)] * n2
        idx = np.flatnonzero(rest)[ok2]
        out[idx] = s2[idx]
    return out


def score(b: pd.DataFrame, out: np.ndarray, ents: np.ndarray, gt_s1: np.ndarray) -> dict:
    m = (out >= 0) & np.isin(out, ents)
    idx = np.searchsorted(ents, out[m])
    tp = (out[m] == b["true_s1"].to_numpy()[m]).astype(float)
    return macro_f05_counts(ntrue_for(ents, gt_s1), np.bincount(idx, minlength=len(ents)),
                            np.bincount(idx, weights=tp, minlength=len(ents)))


def baseline(t: float) -> dict:
    return {"floor": t, "solo": 1.01, "T": [0.0] * len(BUCKETS)}


def tune(b: pd.DataFrame, ents: np.ndarray, gt_s1: np.ndarray, t0: float, passes: int = 3) -> dict:
    p = baseline(t0)
    best = score(b, assign(b, p), ents, gt_s1)["macro_f05"]
    print(f"tune start (baseline t={t0}): {best:.5f}")
    for it in range(passes):
        improved = False
        for name, grid in [("floor", FLOOR_GRID), ("solo", SOLO_GRID)] + [(j, T_GRID) for j in range(len(BUCKETS))]:
            for v in grid:
                q = json.loads(json.dumps(p))
                if isinstance(name, int):
                    q["T"][name] = v
                else:
                    q[name] = v
                f = score(b, assign(b, q), ents, gt_s1)["macro_f05"]
                if f > best + 1e-6:
                    best, p, improved = f, q, True
        print(f"  pass {it + 1}: {best:.5f} {json.dumps(p)}")
        if not improved:
            break
    return p


def slices(b: pd.DataFrame, out: np.ndarray, ents: np.ndarray) -> pd.DataFrame:
    ri = rec_info(b["key"].to_numpy()).reindex(b["key"])
    indic = np.array([has_indic(x) for x in ri["name"].fillna("")])
    empty = ri["addr_empty"].fillna(False).to_numpy().astype(bool)
    core = ri["core"].fillna("").to_numpy()
    pm = (out >= 0) & np.isin(out, ents)
    rows = []
    for kind in ("P", "R"):
        if kind == "P":
            sel, target = pm, out
            hit = out == b["true_s1"].to_numpy()
        else:
            sel = np.isin(b["true_s1"].to_numpy(), ents)
            target = b["true_s1"].to_numpy()
            hit = out == target
        sc = s1_info(target[sel]).reindex(target[sel])["core"].fillna("").tolist()
        sim = np.full(len(b), 100.0, np.float32)
        sim[sel] = cpdist(list(core[sel]), sc, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        sl = np.select([indic, empty, sim < 50], ["Indic", "empty address", "DBA (name_sim<50)"], "other")
        d = pd.DataFrame({"country": b["country"].to_numpy()[sel], "slice": sl[sel], "hit": hit[sel]})
        r = d.groupby(["country", "slice"])["hit"].agg(["mean", "size"]).rename(
            columns={"mean": kind, "size": f"n_{kind}"})
        rows.append(r)
    return rows[0].join(rows[1], how="outer")


def evaluate(a) -> None:
    val_ids, val_cty, val_removed, gt_s1 = val_entities()
    half = (mix64(val_ids) & np.uint64(1)).astype(bool)
    bdir = Path(a.best_dir) if a.best_dir else WORK
    bv = pd.read_parquet(bdir / "best_val.parquet")
    bs = pd.read_parquet(bdir / "best_stress.parquet")
    assert {"s1_2", "prob_2"} <= set(bv.columns), "best_val.parquet has no second-best columns; rerun matcher.py eval"
    base = baseline(a.t)
    p = tune(bv, val_ids[half], gt_s1, a.t)
    PARAMS.write_text(json.dumps(p), encoding="utf-8")
    rows = []
    for part, b, ids in (("val", bv, val_ids), ("stress", bs, val_ids[~val_removed])):
        hs = (mix64(ids) & np.uint64(1)).astype(bool)
        for name, q in (("baseline", base), ("group", p)):
            out = assign(b, q)
            for sub, e in (("all", ids), ("A (tune)", ids[hs]), ("B (held out)", ids[~hs])):
                r = score(b, out, e, gt_s1)
                rows.append({"part": part, "rule": name, "S1 set": sub, "P": r["micro_precision"],
                             "R": r["micro_recall"], "F0.5": r["macro_f05"]})
    pd.set_option("display.width", 200)
    print(f"\nchosen: {json.dumps(p)}")
    print(pd.DataFrame(rows).set_index(["part", "S1 set", "rule"]).round(5).to_string())
    for name, q in (("baseline", base), ("group", p)):
        out = assign(bv, q)
        for c in np.unique(val_cty):
            r = score(bv, out, val_ids[val_cty == c], gt_s1)
            print(f"  val {c} {name}: P {r['micro_precision']:.4f} R {r['micro_recall']:.4f} F0.5 {r['macro_f05']:.4f}")
    sb = slices(bv, assign(bv, base), val_ids)
    sg = slices(bv, assign(bv, p), val_ids)
    t = sb.join(sg, lsuffix="_base", rsuffix="_group")
    print("\nper country x slice (P over predictions, R over GT pairs whose record is scored):")
    print(t[["P_base", "P_group", "R_base", "R_group", "n_P_group", "n_R_group"]].round(4).to_string())


def submit(a) -> None:
    p = json.loads(PARAMS.read_text(encoding="utf-8"))
    pred = pd.read_parquet(Path(a.best_dir) / "test_pred.parquet" if a.best_dir else WORK / "test_pred.parquet")
    assert {"s1_2", "prob_2"} <= set(pred.columns), "test_pred.parquet has no second-best columns"
    out = assign(pred, p)
    m = pred.assign(s1=out)[out >= 0]
    s1 = pd.read_parquet(WORK / "norm" / "test_S1.parquet", columns=["i"])["i"].to_numpy()
    path = OUT / (a.out or "matching_results.tsv")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(pd.DataFrame({"source1_entity_id": "S1-" + pd.Series(s1).astype(str),
                            "matched_entity_ids": join_ids(s1, m)}), path)
    size = path.stat().st_size
    print(f"group assignment {json.dumps(p)}: {len(m):,} matches over {m['s1'].nunique():,} S1; "
          f"{path.name} {size:,} bytes; rerouted to second-best: {int((out == pred['s1_2'].to_numpy()).sum()):,}")
    print(m.groupby("country").size().to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "submit"])
    ap.add_argument("--t", type=float, default=0.71, help="baseline per-record threshold")
    ap.add_argument("--out", help="submit: matching file path relative to output/")
    ap.add_argument("--best-dir", help="directory with best_val/best_stress/test_pred parquet (default work/)")
    a = ap.parse_args()
    {"eval": evaluate, "submit": submit}[a.cmd](a)


if __name__ == "__main__":
    main()
