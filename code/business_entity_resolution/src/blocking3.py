"""Blocking v2 (D-015, D-017): the candidate generator used by the final submission (sub-04b).

Per country, the union of three retrievals in the blocking v1 TF-IDF spaces:
  rank_v1  record -> S1 top-10, re-queried with the dictionary-mapped name for Indic records
  rank_a   address-only record -> S1 top-3
  rank_s   S1-side, S1 -> record top-10 (recovers records crowded out of their own top-k)
Search indexes are hash-permuted (splitmix64) and per-record cosine ties are broken by a pair
hash, so the candidate set never depends on ID order. Pruning happens later in features.py.

Optional add-ons, off in the final pipeline:
  --add-name   exact name-only top-5 for empty-address records (sub-06, D-019; use with ER_KN=5)
  --add-dense  dense e5 top-K pairs from work/dense (sub-07, abandoned; use with ER_KD)

Output: work/blocks3/{split}_{country}.parquet sorted by record, cos descending.
Run: python src/blocking3.py --split train|test
"""

from __future__ import annotations

import argparse
import gc
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

from blocking import Featurizer, countries, topk_rows
from blocking2 import rowdot
from common import WORK, MemTracker
from features import rec_key
from translit import has_indic

K_V1, K_S1, K_A, K_N, K_D, N_RETR, NONE = 10, 10, 3, 5, 5, 50, 127


def load_recs(split: str, c: str) -> pd.DataFrame:
    out = []
    for k in (2, 3):
        t = pd.read_parquet(WORK / "norm" / f"{split}_S{k}.parquet", columns=["i", "name", "core", "addr"],
                            filters=[("country", "==", c)])
        t["key"] = rec_key(np.int8(k), t.pop("i").to_numpy())
        out.append(t)
    return pd.concat(out, ignore_index=True).sort_values("key", ignore_index=True)


def mix64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.int64).astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def hperm(ids: np.ndarray) -> np.ndarray:
    return np.argsort(mix64(ids), kind="stable")


def two_stage_perm(Qr, Qf, Ir, If, perm, K, n_threads, chunk=20_000):
    r, col, cos, rank = two_stage(Qr, Qf, Ir[perm].T.tocsr(), If[perm], K, n_threads, chunk)
    return r, perm[col], cos, rank


def two_stage(Qr, Qf, Ir_T, If, K, n_threads, chunk=20_000):
    R, C, V = [], [], []
    for lo in range(0, Qr.shape[0], chunk):
        m = sp_matmul_topn(Qr[lo:lo + chunk], Ir_T, top_n=N_RETR, n_threads=n_threads).tocsr()
        rows = np.repeat(np.arange(m.shape[0]), np.diff(m.indptr))
        cols = m.indices.astype(np.int64)
        cos = rowdot(Qf[lo:lo + chunk], If, rows, cols)
        keep, rank = topk_rows(rows, cos, K)
        R.append(lo + rows[keep])
        C.append(cols[keep])
        V.append(np.c_[cos[keep], rank])
    v = np.concatenate(V)
    return np.concatenate(R), np.concatenate(C), v[:, 0].astype(np.float32), v[:, 1].astype(np.int8)


def block_country(split: str, c: str, n_threads: int, s1_limit: int, mt: MemTracker) -> pd.DataFrame:
    s1 = pd.read_parquet(WORK / "norm" / f"{split}_S1.parquet", columns=["i", "core", "addr"],
                         filters=[("country", "==", c)]).sort_values("i", ignore_index=True)
    s1_ids = s1["i"].to_numpy()
    full, retr = Featurizer(1.0, 0.05), Featurizer(1.0, 5000)
    xs = full.fit_transform(s1)
    xs_r = retr.fit_transform(s1)
    xs_a = normalize(full.av.transform(s1["addr"].tolist()))
    xs_ar = normalize(retr.av.transform(s1["addr"].tolist()))
    del s1
    recs = load_recs(split, c)
    rkeys = recs["key"].to_numpy()
    xr = full.transform(recs)
    xr_r = retr.transform(recs)
    has_addr = (recs["addr"].str.len() > 0).to_numpy()
    xr_a = normalize(full.av.transform(recs["addr"].tolist()))
    xr_ar = normalize(retr.av.transform(recs["addr"].tolist()))
    indic = np.fromiter((has_indic(x) for x in recs["name"]), bool, len(recs))
    del recs
    gc.collect()
    ps, pr = hperm(s1_ids), hperm(rkeys)
    mt.stage(f"{split} {c}: vectors S1 {len(s1_ids):,}, records {len(rkeys):,} ({int(indic.sum()):,} Indic)")

    v1 = pd.read_parquet(WORK / "blocks" / f"{split}_{c}_K10.parquet", columns=["src", "rec", "s1", "rank"])
    v1["key"] = rec_key(v1["src"].to_numpy(), v1["rec"].to_numpy())
    v1 = v1[~np.isin(v1["key"].to_numpy(), rkeys[indic])][["key", "s1", "rank"]]
    qi = np.flatnonzero(indic)
    if len(qi):
        r, col, _, rank = two_stage_perm(xr_r[qi], xr[qi], xs_r, xs, ps, K_V1, n_threads)
        v1 = pd.concat([v1, pd.DataFrame({"key": rkeys[qi[r]], "s1": s1_ids[col], "rank": rank})],
                       ignore_index=True)
    v1 = v1.rename(columns={"rank": "rank_v1"})
    mt.stage(f"{split} {c}: v1 ({len(v1):,} pairs; {len(qi):,} Indic records re-queried)")

    qa = np.flatnonzero(has_addr)
    r, col, _, rank = two_stage_perm(xr_ar[qa], xr_a[qa], xs_ar, xs_a, ps, K_A, n_threads)
    ad = pd.DataFrame({"key": rkeys[qa[r]], "s1": s1_ids[col], "rank_a": rank})
    del xr_a, xr_ar, xs_a, xs_ar
    gc.collect()
    mt.stage(f"{split} {c}: address top-{K_A} ({len(ad):,} pairs)")

    qs = np.arange(len(s1_ids)) if not s1_limit else np.sort(
        np.random.default_rng(0).choice(len(s1_ids), min(s1_limit, len(s1_ids)), replace=False))
    t0 = time.perf_counter()
    r, col, _, rank = two_stage_perm(xs_r[qs], xs[qs], xr_r, xr, pr, K_S1, n_threads, chunk=5_000)
    sd = pd.DataFrame({"key": rkeys[col], "s1": s1_ids[qs[r]], "rank_s": rank})
    mt.stage(f"{split} {c}: S1-side top-{K_S1} for {len(qs):,} S1 ({len(sd):,} pairs, "
             f"{(time.perf_counter() - t0) / len(qs) * 1e3:.2f} ms/S1)")

    ns = np.int64(len(s1_ids))
    parts = []
    for j, d in enumerate((v1, ad, sd)):
        pid = np.searchsorted(rkeys, d["key"].to_numpy()).astype(np.int64) * ns + np.searchsorted(s1_ids, d["s1"].to_numpy())
        parts.append((pid, d.iloc[:, 2].to_numpy().astype(np.int8), j))
    del v1, ad, sd
    gc.collect()
    pid, inv = np.unique(np.concatenate([p[0] for p in parts]), return_inverse=True)
    ranks = np.full((len(pid), 3), NONE, np.int8)
    off = 0
    for p_, rk, j in parts:
        np.minimum.at(ranks[:, j], inv[off:off + len(p_)], rk)
        off += len(p_)
    del parts, inv
    ri, si = pid // ns, pid % ns
    del pid
    cos = np.empty(len(ri), np.float32)
    for lo in range(0, len(ri), 5_000_000):
        cos[lo:lo + 5_000_000] = rowdot(xr, xs, ri[lo:lo + 5_000_000], si[lo:lo + 5_000_000])
    key = rkeys[ri]
    o = np.lexsort((mix64(key) ^ mix64(s1_ids[si]), -cos, key))
    u = pd.DataFrame({"src": (key[o] // 10**12).astype(np.int8), "rec": key[o] % 10**12, "s1": s1_ids[si[o]],
                      "cos": cos[o], "rank_v1": ranks[o, 0], "rank_a": ranks[o, 1], "rank_s": ranks[o, 2]})
    mt.stage(f"{split} {c}: union {len(u):,} pairs")
    return u


def add_name(split: str, c: str, n_threads: int, mt: MemTracker) -> pd.DataFrame:
    path = WORK / "blocks3" / f"{split}_{c}.parquet"
    u = pd.read_parquet(path)
    if "rank_n" in u.columns:
        other = u[["rank_v1", "rank_a", "rank_s"]].to_numpy().astype(np.int16).min(axis=1) < NONE
        u = u[other].drop(columns="rank_n")
    s1 = pd.read_parquet(WORK / "norm" / f"{split}_S1.parquet", columns=["i", "core", "addr"],
                         filters=[("country", "==", c)]).sort_values("i", ignore_index=True)
    s1_ids = s1["i"].to_numpy()
    full = Featurizer(1.0, 0.05)
    xs = full.fit_transform(s1)
    xs_n = normalize(full.nv.transform(s1["core"].tolist()))
    del s1
    recs = load_recs(split, c)
    e = recs[(recs["addr"].str.len() == 0).to_numpy()].reset_index(drop=True)
    del recs
    ekeys = e["key"].to_numpy()
    xe = full.transform(e)
    xe_n = normalize(full.nv.transform(e["core"].tolist()))
    del e
    ps = hperm(s1_ids)
    bt = xs_n[ps].T.tocsr()
    R, C, V = [], [], []
    for lo in range(0, xe_n.shape[0], 5_000):
        m = sp_matmul_topn(xe_n[lo:lo + 5_000], bt, top_n=K_N, n_threads=n_threads).tocsr()
        R.append(lo + np.repeat(np.arange(m.shape[0]), np.diff(m.indptr)))
        C.append(ps[m.indices.astype(np.int64)])
        V.append(m.data.astype(np.float32))
    r, col, v = np.concatenate(R), np.concatenate(C), np.concatenate(V)
    keep, rank = topk_rows(r, v, K_N)
    r, col, rank = r[keep], col[keep], rank[keep]
    mt.stage(f"{split} {c}: exact name-only top-{K_N} for {len(ekeys):,} empty-address records ({len(r):,} pairs)")
    new = pd.DataFrame({"key": ekeys[r], "s1": s1_ids[col], "rank_n": rank.astype(np.int8),
                        "cos_new": rowdot(xe, xs, r, col).astype(np.float32)})
    ukey = rec_key(u["src"].to_numpy(), u["rec"].to_numpy())
    em = np.isin(ukey, ekeys)
    ue = u[em].assign(key=ukey[em])
    rest = u[~em].assign(rank_n=np.int8(NONE))
    del u
    ue = ue.merge(new, on=["key", "s1"], how="outer")
    for x in ("rank_v1", "rank_a", "rank_s", "rank_n"):
        ue[x] = ue[x].fillna(NONE).astype(np.int8)
    ue["cos"] = ue["cos"].fillna(ue["cos_new"]).astype(np.float32)
    ue["src"] = (ue["key"].to_numpy() // 10**12).astype(np.int8)
    ue["rec"] = ue["key"].to_numpy() % 10**12
    k = ue["key"].to_numpy()
    o = np.lexsort((mix64(k) ^ mix64(ue["s1"].to_numpy()), -ue["cos"].to_numpy(), k))
    ue = ue.iloc[o][rest.columns]
    out = pd.concat([rest, ue], ignore_index=True)
    del rest, ue
    o = np.argsort(rec_key(out["src"].to_numpy(), out["rec"].to_numpy()), kind="stable")
    out = out.iloc[o].reset_index(drop=True)
    n_new = int((out["rank_n"].to_numpy() < NONE).sum())
    mt.stage(f"{split} {c}: union {len(out):,} pairs ({n_new:,} with rank_n)")
    return out


def add_dense(split: str, c: str, mt: MemTracker, n_chunks: int = 40) -> None:
    path = WORK / "blocks3" / f"{split}_{c}.parquet"
    u = pq.read_table(path).to_pandas()
    if "rank_d" in u.columns:
        other = u[[x for x in ("rank_v1", "rank_a", "rank_s", "rank_n") if x in u.columns]].to_numpy().astype(np.int16).min(axis=1) < NONE
        u = u[other].drop(columns="rank_d")
    rcols = [x for x in ("rank_v1", "rank_a", "rank_s", "rank_n") if x in u.columns]
    ukey = rec_key(u["src"].to_numpy(), u["rec"].to_numpy())
    z = np.load(WORK / "dense" / f"{split}_{c}_pairs.npz")
    kd = min(K_D, z["s1_id"].shape[1])
    dkey = np.repeat(z["record_id"], kd)
    ds1 = np.ascontiguousarray(z["s1_id"][:, :kd]).ravel()
    drank = np.tile(np.arange(kd, dtype=np.int8), len(z["record_id"]))
    del z
    s1 = pd.read_parquet(WORK / "norm" / f"{split}_S1.parquet", columns=["i", "core", "addr"],
                         filters=[("country", "==", c)]).sort_values("i", ignore_index=True)
    s1_ids = s1["i"].to_numpy()
    full = Featurizer(1.0, 0.05)
    xs = full.fit_transform(s1)
    del s1
    recs = load_recs(split, c)
    rkeys = recs["key"].to_numpy()
    xr = full.transform(recs)
    del recs
    gc.collect()
    mt.stage(f"{split} {c}: union {len(u):,} pairs, dense rank<{K_D} {len(dkey):,} pairs, vectors ready")
    bounds = np.r_[rkeys[::max(1, len(rkeys) // n_chunks)], rkeys[-1] + 1]
    tmp = path.with_suffix(".tmp.parquet")
    w, n_out, n_new = None, 0, 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        us = slice(*np.searchsorted(ukey, [a, b]))
        dsl = slice(*np.searchsorted(dkey, [a, b]))
        uc = u.iloc[us].assign(key=ukey[us])
        dc = pd.DataFrame({"key": dkey[dsl], "s1": ds1[dsl], "rank_d": drank[dsl]})
        mc = uc.merge(dc, on=["key", "s1"], how="outer")
        for x in rcols + ["rank_d"]:
            mc[x] = mc[x].fillna(NONE).astype(np.int8)
        new = mc["cos"].isna().to_numpy()
        if new.any():
            ri = np.searchsorted(rkeys, mc["key"].to_numpy()[new])
            si = np.searchsorted(s1_ids, mc["s1"].to_numpy()[new])
            mc.loc[new, "cos"] = rowdot(xr, xs, ri, si)
        n_new += int(new.sum())
        k = mc["key"].to_numpy()
        mc["src"] = (k // 10**12).astype(np.int8)
        mc["rec"] = k % 10**12
        mc["cos"] = mc["cos"].astype(np.float32)
        o = np.lexsort((mix64(k) ^ mix64(mc["s1"].to_numpy()), -mc["cos"].to_numpy(), k))
        t = pa.Table.from_pandas(mc.iloc[o][["src", "rec", "s1", "cos"] + rcols + ["rank_d"]], preserve_index=False)
        w = w or pq.ParquetWriter(tmp, t.schema)
        w.write_table(t)
        n_out += len(mc)
    w.close()
    tmp.replace(path)
    mt.stage(f"{split} {c}: union {n_out:,} pairs ({n_new:,} new from dense)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--countries", nargs="*")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--s1-limit", type=int, default=0, help="bench: S1-side queries for a random S1 sample only")
    ap.add_argument("--tag", default="")
    ap.add_argument("--add-name", action="store_true", help="add exact name-only top-5 for empty-address records to existing blocks3")
    ap.add_argument("--add-dense", action="store_true", help="add dense e5 top-K_D pairs (work/dense) to existing blocks3")
    a = ap.parse_args()
    mt = MemTracker()
    (WORK / "blocks3").mkdir(parents=True, exist_ok=True)
    for c in a.countries or countries(a.split):
        if a.add_dense:
            add_dense(a.split, c, mt)
            continue
        u = add_name(a.split, c, a.threads, mt) if a.add_name else block_country(a.split, c, a.threads, a.s1_limit, mt)
        path = WORK / "blocks3" / f"{a.split}_{c}{a.tag}.parquet"
        tmp = path.with_suffix(".tmp.parquet")
        u.to_parquet(tmp, index=False)
        tmp.replace(path)
        print(f"{a.split} {c}: {len(u):,} union pairs")
        del u
        gc.collect()


if __name__ == "__main__":
    main()
