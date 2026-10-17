"""LightGBM pairwise matcher: features, training, evaluation, audit, inference and output.

Decision rule: for each S2/S3 record take its max-probability S1 candidate and assign it if
prob >= t, which enforces at most one S1 per record.

  build     train split -> work/feats/{fit,val,stress}_{country}.parquet
            fit    = records with no val S1 among their candidates and true S1 not in val
            val    = records touching a sampled val S1 (exact per-S1 attribution)
            stress = val with 20% of the sampled S1 removed from all candidate lists
  train     LightGBM; early stopping on a record-grouped 10% of fit, never on val
  eval      val / stress macro F0.5 per threshold, per country, FP breakdown
  features  leakage audit: no ID / position / GT / country columns, Spearman vs IDs
  predict   test inference in record batches -> work/test_pred*.parquet
  report    test probability distribution and match rate per country
  submit    --t -> output/matching_results.tsv and output/candidate_pairs.tsv
  size_t, budget  sub-02b / D-012 output-size experiments (not used in the final run)
"""

from __future__ import annotations

import argparse
import os
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from blocking import countries
from common import OUT, SEED, WORK, MemTracker, is_val, write_tsv
from features import BASE_COLS, GRAPH_COLS, SIB_COLS, Country, rec_key
from metrics import macro_f05_counts

FEATS = WORK / "feats"
META = ["label", "key", "rec", "s1", "true_s1", "country"]
N_FIT = 1_500_000
N_VAL = int(os.environ.get("ER_NVAL", 100_000))
TRAIN_RECORDS = 10_320_219
GRID = np.round(np.arange(0.05, 0.99, 0.01), 3)


def gt_lookup():
    gt = pd.read_parquet(WORK / "norm" / "train_gt.parquet")
    k = rec_key(gt["src"], gt["rec"])
    o = np.argsort(k)
    return k[o], gt["s1"].to_numpy()[o]


def true_s1_of(keys: np.ndarray, gk: np.ndarray, gs: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(gk, keys).clip(max=len(gk) - 1)
    return np.where(gk[pos] == keys, gs[pos], -1)


def write_feats(ctx: Country, cand: pd.DataFrame, path, extra: dict, reuse=None) -> int:
    w, n = None, 0
    for i, f in enumerate(ctx.features(cand, reuse=reuse)):
        lo = i * 2_000_000
        for k, v in extra.items():
            f[k] = v[lo:lo + len(f)]
        f["key"] = cand["key"].to_numpy()[lo:lo + len(f)]
        t = pa.Table.from_pandas(f, preserve_index=False)
        w = w or pq.ParquetWriter(path, t.schema)
        w.write_table(t)
        n += len(f)
    if w:
        w.close()
    return n


def build(a) -> None:
    mt = MemTracker()
    FEATS.mkdir(parents=True, exist_ok=True)
    gk, gs = gt_lookup()
    for c in a.countries or countries("train"):
        ctx = Country("train", c)
        cand = ctx.candidates()
        ctx.save_graph(FEATS / f"graph_train_{c}.parquet")
        mt.stage(f"{c}: S1 texts + tfidf, candidates + graph ({len(cand):,} pairs)")
        keys, s1 = cand["key"].to_numpy(), cand["s1"].to_numpy()
        ts = true_s1_of(keys, gk, gs)
        val_c = ctx.S.ids[is_val(ctx.S.ids)]
        samp = np.sort(np.random.default_rng(SEED).choice(val_c, min(N_VAL, len(val_c)), replace=False))
        removed = np.sort(np.random.default_rng(SEED + 1).choice(samp, int(0.2 * len(samp)), replace=False))
        pd.DataFrame({"s1": samp, "removed": np.isin(samp, removed)}).to_parquet(
            FEATS / f"val_sample_{c}.parquet", index=False)
        val_keys = np.unique(keys[np.isin(s1, samp) | np.isin(ts, samp)])
        any_val = np.unique(keys[is_val(s1) | ((ts >= 0) & is_val(np.maximum(ts, 0)))])
        pool = np.setdiff1d(np.unique(keys), any_val)
        n_fit = int(N_FIT * len(ctx.r_key) / TRAIN_RECORDS)
        fit_keys = np.random.default_rng(SEED).choice(pool, size=min(n_fit, len(pool)), replace=False)
        for part, k in (("fit", fit_keys), ("val", val_keys)):
            if part not in a.parts:
                continue
            m = np.isin(keys, k)
            cs = cand[m].reset_index(drop=True)
            t = ts[m]
            n = write_feats(ctx, cs, FEATS / f"{part}_{c}.parquet",
                            {"label": cs["s1"].to_numpy() == t, "true_s1": t, "country": np.full(len(cs), c)})
            mt.stage(f"{c}: {part} {len(k):,} records, {n:,} pairs")
            del cs
        del cand, keys, s1, ts
        if "stress" not in a.parts:
            continue
        cand2 = ctx.candidates(remove_s1=removed)
        cs = cand2[np.isin(cand2["key"].to_numpy(), val_keys)].reset_index(drop=True)
        del cand2
        pair_cols = [x for x in pq.read_schema(FEATS / f"val_{c}.parquet").names
                     if x not in META and x not in GRAPH_COLS + SIB_COLS + BASE_COLS]
        reuse = pd.read_parquet(FEATS / f"val_{c}.parquet", columns=["key", "s1"] + pair_cols)
        reuse = reuse[np.isin(reuse["key"].to_numpy(), cs["key"].to_numpy())]
        t = true_s1_of(cs["key"].to_numpy(), gk, gs)
        n = write_feats(ctx, cs, FEATS / f"stress_{c}.parquet",
                        {"label": cs["s1"].to_numpy() == t, "true_s1": t, "country": np.full(len(cs), c)},
                        reuse=reuse)
        n_new = len(cs) - len(cs[["key", "s1"]].merge(reuse[["key", "s1"]], on=["key", "s1"]))
        mt.stage(f"{c}: stress {cs['key'].nunique():,} records, {n:,} pairs ({n_new:,} new pairs)")
        del ctx, cs, reuse


def load(part: str) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(FEATS / f"{part}_{c}.parquet") for c in countries("train")],
                     ignore_index=True)


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META]


def train(a) -> None:
    df = load("fit")
    cols = feature_cols(df)
    X = df[cols].astype(np.float32)
    y = df["label"].to_numpy()
    hold = is_val(df["key"].to_numpy(), 10)
    print(f"fit: {len(df):,} pairs, {df['key'].nunique():,} records, pos rate {y.mean():.3f}, "
          f"{len(cols)} features")
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=8, seed=SEED, verbose=-1)
    dtr = lgb.Dataset(X[~hold], y[~hold])
    dho = lgb.Dataset(X[hold], y[hold], reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dho],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    m.save_model(str(WORK / "matcher.txt"))
    imp = pd.DataFrame({"feature": cols, "gain": m.feature_importance("gain"),
                        "split": m.feature_importance("split")}).sort_values("gain", ascending=False)
    imp["gain_share"] = (imp["gain"] / imp["gain"].sum()).round(4)
    imp.to_csv(WORK / "matcher_importance.tsv", sep="\t", index=False, lineterminator="\n")
    print(imp.head(30).to_string(index=False))


def top2(d: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    d = d.sort_values(by + ["prob"], ascending=[True] * len(by) + [False]).drop_duplicates(by + ["s1"])
    r = d.groupby(by, sort=False).cumcount().to_numpy()
    b1 = d[r == 0].reset_index(drop=True)
    b2 = d.loc[r == 1, by + ["s1", "prob"]].rename(columns={"s1": "s1_2", "prob": "prob_2"})
    b1 = b1.merge(b2, on=by, how="left")
    b1["s1_2"] = b1["s1_2"].fillna(-1).astype(np.int64)
    b1["prob_2"] = b1["prob_2"].fillna(0).astype(np.float32)
    return b1


def best_per_record(df: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    d = pd.DataFrame({"key": df["key"].to_numpy(), "s1": df["s1"].to_numpy(), "prob": prob,
                      "label": df["label"].to_numpy() if "label" in df else False,
                      "true_s1": df["true_s1"].to_numpy() if "true_s1" in df else -1,
                      "country": df["country"].to_numpy()})
    return top2(d, ["key"])


def score_curve(best: pd.DataFrame, ents: np.ndarray, ntrue: np.ndarray, name: str) -> pd.DataFrame:
    best = best[np.isin(best["s1"].to_numpy(), ents)]
    idx = np.searchsorted(ents, best["s1"].to_numpy())
    prob, lab = best["prob"].to_numpy(), best["label"].to_numpy().astype(np.float64)
    rows = []
    for t in GRID:
        m = prob >= t
        r = macro_f05_counts(ntrue, np.bincount(idx[m], minlength=len(ents)),
                             np.bincount(idx[m], weights=lab[m], minlength=len(ents)))
        rows.append({"t": t, f"{name}_P": r["micro_precision"], f"{name}_R": r["micro_recall"],
                     f"{name}_F05": r["macro_f05"]})
    return pd.DataFrame(rows).set_index("t")


def val_entities():
    v = pd.concat([pd.read_parquet(FEATS / f"val_sample_{c}.parquet").assign(country=c)
                   for c in countries("train")]).sort_values("s1")
    gt = pd.read_parquet(WORK / "norm" / "train_gt.parquet", columns=["s1"])["s1"].to_numpy()
    return v["s1"].to_numpy(), v["country"].to_numpy(), v["removed"].to_numpy(), gt


def ntrue_for(ents: np.ndarray, gt_s1: np.ndarray) -> np.ndarray:
    g = gt_s1[np.isin(gt_s1, ents)]
    return np.bincount(np.searchsorted(ents, g), minlength=len(ents))


def evaluate(a) -> None:
    m = lgb.Booster(model_file=str(WORK / "matcher.txt"))
    val_ids, val_cty, val_removed, gt_s1 = val_entities()
    kept = val_ids[~val_removed]
    curves, bests = [], {}
    for part, ents in (("val", val_ids), ("stress", kept)):
        df = load(part)
        prob = m.predict(df[m.feature_name()].astype(np.float32), num_threads=8)
        print(f"{part}: {len(df):,} pairs, {df['key'].nunique():,} records")
        bests[part] = best_per_record(df, prob)
        bests[part].to_parquet(WORK / f"best_{part}.parquet", index=False)
        curves.append(score_curve(bests[part], ents, ntrue_for(ents, gt_s1), part))
        del df
    cur = curves[0].join(curves[1])
    cur.to_csv(WORK / "matcher_curve.tsv", sep="\t", lineterminator="\n")
    pd.set_option("display.width", 200)
    print(cur.iloc[::5].round(4).to_string())
    t = a.t if a.t is not None else float(cur["val_F05"].idxmax())
    print(f"\nat t={t}: " + json.dumps(cur.loc[t].round(4).to_dict()))
    bv = bests["val"]
    for c in np.unique(val_cty):
        ents = val_ids[val_cty == c]
        r = score_curve(bv, ents, ntrue_for(ents, gt_s1), "v").loc[t]
        print(f"  val {c}: P {r['v_P']:.4f} R {r['v_R']:.4f} F0.5 {r['v_F05']:.4f}")
    p = bv[(bv["prob"] >= t) & np.isin(bv["s1"].to_numpy(), val_ids)]
    kind = np.select([p["label"], p["true_s1"] < 0], ["TP", "FP(a) distractor"], "FP(b) wrong S1")
    print(pd.Series(kind).value_counts(normalize=True).round(4).to_string())


FORBIDDEN = {"rec", "s1", "key", "r", "s", "true_s1", "label", "country", "i", "row", "pos"}


def check_features(a) -> None:
    m = lgb.Booster(model_file=str(WORK / "matcher.txt"))
    names = m.feature_name()
    bad = sorted(FORBIDDEN & set(names))
    print(f"{len(names)} model features:\n  " + ", ".join(names))
    print(f"forbidden (ID/position/GT) columns in model: {bad or 'none'}")
    assert not bad
    df = pd.read_parquet(FEATS / "val_US.parquet").sample(500_000, random_state=SEED)
    df = pd.concat([df, pd.read_parquet(FEATS / "val_India.parquet").sample(500_000, random_state=SEED)])
    rk = df[names + ["rec", "s1", "label"]].rank()
    corr = pd.DataFrame({"rec_id": rk[names].corrwith(rk["rec"]).abs(),
                         "s1_id": rk[names].corrwith(rk["s1"]).abs()})
    print(f"max |spearman| with rec ID: {corr['rec_id'].max():.4f} ({corr['rec_id'].idxmax()}), "
          f"with S1 ID: {corr['s1_id'].max():.4f} ({corr['s1_id'].idxmax()})")
    print(f"label vs rec ID |spearman| {abs(rk['label'].corr(rk['rec'])):.4f}, "
          f"vs S1 ID {abs(rk['label'].corr(rk['s1'])):.4f}")
    print(f"pairs whose cos ties another candidate (rank order arbitrary): {df['tie'].mean():.4f}")


def predict(a) -> None:
    mt = MemTracker()
    m = lgb.Booster(model_file=str(WORK / "matcher.txt"))
    for c in a.countries or countries("test"):
        done = WORK / f"test_pred_{c}.parquet"
        if done.exists():
            print(f"test {c}: already predicted ({done.name}), skipping")
            continue
        ctx = Country("test", c)
        cand = ctx.candidates()
        ctx.save_graph(FEATS / f"graph_test_{c}.parquet")
        cand[["src", "rec", "s1"]].to_parquet(WORK / f"test_cands_{c}.parquet", index=False)
        keys = cand["key"].to_numpy()
        uk = np.unique(keys)
        prob = np.empty(len(cand), np.float32)
        for lo in range(0, len(uk), a.batch):
            sel = np.flatnonzero(np.isin(keys, uk[lo:lo + a.batch]))
            ps = [m.predict(f[m.feature_name()].astype(np.float32), num_threads=8)
                  for f in ctx.features(cand.iloc[sel].reset_index(drop=True))]
            prob[sel] = np.concatenate(ps)
            mt.stage(f"test {c}: {min(lo + a.batch, len(uk)):,}/{len(uk):,} records")
        d = pd.DataFrame({"src": cand["src"].to_numpy(), "rec": cand["rec"].to_numpy(),
                          "s1": cand["s1"].to_numpy(), "prob": prob, "country": c})
        top2(d, ["src", "rec"]).to_parquet(done, index=False)
        del ctx, cand, d
    cs = countries("test")
    if all((WORK / f"test_pred_{c}.parquet").exists() for c in cs):
        pd.concat([pd.read_parquet(WORK / f"test_pred_{c}.parquet") for c in cs],
                  ignore_index=True).to_parquet(WORK / "test_pred.parquet", index=False)
        pd.concat([pd.read_parquet(WORK / f"test_cands_{c}.parquet") for c in cs],
                  ignore_index=True).to_parquet(WORK / "test_cands.parquet", index=False)


def report(a) -> None:
    p = pd.read_parquet(WORK / "test_pred.parquet", columns=["country", "prob"])
    q = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
    rows = []
    for c, g in p.groupby("country"):
        x = g["prob"].to_numpy()
        r = {"country": c, "records": len(x)}
        r.update({f"q{int(k * 100)}": np.quantile(x, k) for k in q})
        r.update({f"rate@{t}": (x >= t).mean() for t in a.ts})
        rows.append(r)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).set_index("country").round(4).to_string())
    bv = pd.read_parquet(WORK / "best_val.parquet", columns=["country", "prob", "true_s1"])
    for c, g in bv.groupby("country"):
        print(f"  val {c} (reference): " + ", ".join(f"rate@{t} {(g['prob'] >= t).mean():.4f}" for t in a.ts)
              + f"; true matched share {(g['true_s1'] >= 0).mean():.4f}")


def size_threshold(a) -> None:
    s1_all = pd.read_parquet(WORK / "norm" / "test_S1.parquet", columns=["i"])["i"].to_numpy()
    test = pd.read_parquet(WORK / "test_pred.parquet")
    cur = pd.read_csv(WORK / "matcher_curve.tsv", sep="\t", index_col="t")
    for t in GRID:
        b = out_bytes(s1_all, test[test["prob"] >= t])
        if b <= a.max_bytes:
            r = cur.loc[t]
            print(f"t={t}: {b:,} bytes; val F0.5 {r['val_F05']:.4f} (P {r['val_P']:.4f} R {r['val_R']:.4f}), "
                  f"stress F0.5 {r['stress_F05']:.4f}")
            return
    print(f"no t fits {a.max_bytes:,} bytes")


def f05_of_recall(r):
    return 1.25 * r / (0.25 + r)


def select(best: pd.DataFrame, t: float, v: float = 0.0) -> pd.DataFrame:
    d = best[best["prob"] >= t].sort_values(["s1", "prob"], ascending=[True, False])
    j = d.groupby("s1").cumcount().to_numpy() + 1
    n = d.groupby("s1")["prob"].transform("size").to_numpy()
    val = d["prob"].to_numpy() * (f05_of_recall(j / n) - f05_of_recall((j - 1) / n))
    return d[val >= v].assign(value=val[val >= v])


def out_bytes(s1_all: np.ndarray, m: pd.DataFrame) -> int:
    nd = lambda x: pd.Series(x).astype(str).str.len().to_numpy()
    fixed = len("source1_entity_id\tmatched_entity_ids\n") + int((3 + nd(s1_all) + 2).sum())
    return fixed + int((3 + nd(m["rec"].to_numpy())).sum()) + len(m) - m["s1"].nunique()


def budget(a) -> None:
    s1_all = pd.read_parquet(WORK / "norm" / "test_S1.parquet", columns=["i"])["i"].to_numpy()
    test = pd.read_parquet(WORK / "test_pred.parquet")
    full = select(test, a.t)
    vs = np.r_[0.0, np.quantile(full["value"].to_numpy(), np.linspace(0.01, 0.6, 60))]
    rows = []
    for v in vs:
        m = full[full["value"].to_numpy() >= v]
        rows.append({"v": v, "test_matches": len(m), "bytes": out_bytes(s1_all, m)})
    sz = pd.DataFrame(rows)
    ok = sz[sz["bytes"] <= a.max_bytes]
    if not len(ok):
        print(sz.to_string())
        print(f"no v fits {a.max_bytes:,} bytes at t={a.t}")
        return
    v = float(ok["v"].min())
    val_ids, val_cty, val_removed, gt_s1 = val_entities()
    out = []
    for vv in sorted({0.0, v} | set(np.round(vs[::6], 4))):
        r = {"v": vv}
        for part, ents in (("val", val_ids), ("stress", val_ids[~val_removed])):
            b = select(pd.read_parquet(WORK / f"best_{part}.parquet"), a.t, vv)
            b = b[np.isin(b["s1"].to_numpy(), ents)]
            idx = np.searchsorted(ents, b["s1"].to_numpy())
            res = macro_f05_counts(ntrue_for(ents, gt_s1), np.bincount(idx, minlength=len(ents)),
                                   np.bincount(idx, weights=b["label"].to_numpy().astype(float), minlength=len(ents)))
            r.update({f"{part}_P": res["micro_precision"], f"{part}_R": res["micro_recall"],
                      f"{part}_F05": res["macro_f05"]})
        m = full[full["value"].to_numpy() >= vv]
        r.update({"test_matches": len(m), "MB": out_bytes(s1_all, m) / 1e6})
        out.append(r)
    res = pd.DataFrame(out).set_index("v")
    pd.set_option("display.width", 200)
    print(res.round(4).to_string())
    print(f"\nt={a.t}: smallest v with test file <= {a.max_bytes:,} bytes: v={v:.5f}")
    bv = select(pd.read_parquet(WORK / "best_val.parquet"), a.t, v)
    for c in np.unique(val_cty):
        ents = val_ids[val_cty == c]
        b = bv[np.isin(bv["s1"].to_numpy(), ents)]
        idx = np.searchsorted(ents, b["s1"].to_numpy())
        r = macro_f05_counts(ntrue_for(ents, gt_s1), np.bincount(idx, minlength=len(ents)),
                             np.bincount(idx, weights=b["label"].to_numpy().astype(float), minlength=len(ents)))
        print(f"  val {c}: P {r['micro_precision']:.4f} R {r['micro_recall']:.4f} F0.5 {r['macro_f05']:.4f}")
    p = bv[np.isin(bv["s1"].to_numpy(), val_ids)]
    kind = np.select([p["label"], p["true_s1"] < 0], ["TP", "FP(a) distractor"], "FP(b) wrong S1")
    print(pd.Series(kind).value_counts(normalize=True).round(4).to_string())
    res.to_csv(WORK / "budget_curve.tsv", sep="\t", lineterminator="\n")


def join_ids(s1_order: np.ndarray, p: pd.DataFrame) -> np.ndarray:
    ids = "S" + p["src"].astype(str) + "-" + p["rec"].astype(str)
    g = ids.groupby(p["s1"].to_numpy()).agg(",".join)
    return pd.Series(s1_order).map(g).fillna("").to_numpy()


def submit(a) -> None:
    s1 = pd.read_parquet(WORK / "norm" / "test_S1.parquet", columns=["i"])["i"].to_numpy()
    s1_str = "S1-" + pd.Series(s1).astype(str)
    pred = pd.read_parquet(WORK / "test_pred.parquet")
    m = select(pred, a.t, a.v)
    out = OUT / a.out if a.out else OUT / "matching_results.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(pd.DataFrame({"source1_entity_id": s1_str, "matched_entity_ids": join_ids(s1, m)}), out)
    size = out.stat().st_size
    print(f"t={a.t} v={a.v}: {len(m):,} matches over {m['s1'].nunique():,} S1; "
          f"{out.name} {size:,} bytes ({size / 1e6:.2f} MB)")
    print(m.groupby("country").size().to_string())
    if not a.matching_only:
        cand = pd.read_parquet(WORK / "test_cands.parquet")
        write_tsv(pd.DataFrame({"source1_entity_id": s1_str, "candidate_entity_ids": join_ids(s1, cand)}),
                  OUT / "candidate_pairs.tsv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "train", "eval", "features", "predict", "report", "size_t", "budget", "submit"])
    ap.add_argument("--t", type=float)
    ap.add_argument("--v", type=float, default=0.0, help="byte-budget value cut (D-012)")
    ap.add_argument("--max-bytes", type=int, default=90_000_000)
    ap.add_argument("--matching-only", action="store_true")
    ap.add_argument("--out", help="submit: matching file path relative to output/ (default matching_results.tsv)")
    ap.add_argument("--ts", type=float, nargs="*", default=[0.5, 0.69, 0.9], help="report: thresholds")
    ap.add_argument("--countries", nargs="*")
    ap.add_argument("--parts", nargs="*", default=["fit", "val", "stress"])
    ap.add_argument("--batch", type=int, default=500_000, help="predict: records per batch")
    a = ap.parse_args()
    {"build": build, "train": train, "eval": evaluate, "features": check_features, "predict": predict,
     "report": report, "size_t": size_threshold, "budget": budget,
     "submit": submit}[a.cmd](a)


if __name__ == "__main__":
    main()
