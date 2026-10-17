"""EXPERIMENT (D-010, not used in the final pipeline): union of name-only, address-only and
consonant-skeleton retrievals.

The 20k-query bench gave lower recall at higher cost than blocking v1, because generic name
words and the lack of rare n-grams defeat name-only search. Superseded by blocking3.py.
`rowdot` is still imported by blocking3.py.
"""

from __future__ import annotations

import argparse
import gc
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

from blocking import TOKEN, countries
from common import WORK, MemTracker
from translit import has_indic, skeleton

RANK_NONE = 127
VIEWS = ("n", "a", "s")


def _vec(view: str, max_df) -> TfidfVectorizer:
    kw = dict(max_df=max_df, dtype=np.float32, sublinear_tf=True)
    if view == "n":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), **kw)
    if view == "a":
        return TfidfVectorizer(token_pattern=TOKEN, lowercase=False, **kw)
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), **kw)


def load(split: str, src: str, country: str) -> pd.DataFrame:
    df = pd.read_parquet(WORK / "norm" / f"{split}_{src}.parquet", columns=["i", "name", "core", "addr"],
                         filters=[("country", "==", country)])
    ind = np.fromiter((has_indic(x) for x in df["name"].tolist()), bool, len(df))
    return df.assign(skel=[skeleton(x) for x in df["core"].tolist()], indic=ind).drop(columns="name")


COL = {"n": "core", "a": "addr", "s": "skel"}


class View:
    def __init__(self, view: str, s1: pd.DataFrame, max_df: float, retr_max_df: int):
        self.v = view
        self.full, self.retr = _vec(view, max_df), _vec(view, retr_max_df)
        txt = s1[COL[view]].tolist()
        self.x1 = normalize(self.full.fit_transform(txt))
        self.x1r_t = normalize(self.retr.fit_transform(txt)).T.tocsr()

    def q_full(self, q: pd.DataFrame) -> sp.csr_matrix:
        return normalize(self.full.transform(q[COL[self.v]].tolist()))

    def search(self, q: pd.DataFrame, qf: sp.csr_matrix, K: int, N: int, n_threads: int):
        r = sp_matmul_topn(normalize(self.retr.transform(q[COL[self.v]].tolist())), self.x1r_t,
                           top_n=N, n_threads=n_threads).tocsr()
        rows = np.repeat(np.arange(len(q)), np.diff(r.indptr))
        cols = r.indices
        cos = rowdot(qf, self.x1, rows, cols)
        keep, rank = topk_rows(rows, cos, K)
        return rows[keep], cols[keep], rank


def rowdot(a: sp.csr_matrix, b: sp.csr_matrix, ra: np.ndarray, rb: np.ndarray, step: int = 2_000_000):
    out = np.empty(len(ra), np.float32)
    for s in range(0, len(ra), step):
        e = s + step
        out[s:e] = np.asarray(a[ra[s:e]].multiply(b[rb[s:e]]).sum(axis=1)).ravel()
    return out


def topk_rows(rows: np.ndarray, score: np.ndarray, K: int):
    o = np.lexsort((-score, rows))
    r = rows[o]
    start = np.r_[0, np.flatnonzero(np.diff(r)) + 1]
    rank = np.arange(len(r)) - np.repeat(start, np.diff(np.r_[start, len(r)]))
    k = rank < K
    return o[k], rank[k]


def block_chunk(views: dict, s1_ids, qc: pd.DataFrame, K: dict, N: int, n_threads: int) -> pd.DataFrame:
    qf = {v: views[v].q_full(qc) for v in VIEWS}
    found = []
    for v in VIEWS:
        mask = np.ones(len(qc), bool)
        if v == "a":
            mask = (qc["addr"] != "").to_numpy()
        elif v == "s":
            mask = qc["indic"].to_numpy()
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue
        rows, cols, rank = views[v].search(qc.iloc[idx], qf[v][idx], K[v], N, n_threads)
        found.append(pd.DataFrame({"q": idx[rows], "c": cols, "view": v, "rank": rank.astype(np.int8)}))
    f = pd.concat(found, ignore_index=True)
    ranks = f.pivot_table(index=["q", "c"], columns="view", values="rank", aggfunc="min")
    ranks = ranks.reindex(columns=list(VIEWS)).fillna(RANK_NONE).astype(np.int8).reset_index()
    q, c = ranks["q"].to_numpy(), ranks["c"].to_numpy()
    out = pd.DataFrame({"q": q, "s1": s1_ids[c]})
    for v in VIEWS:
        out[f"cos_{v}"] = rowdot(qf[v], views[v].x1, q, c)
    qn = 1 + (qc["addr"] != "").to_numpy().astype(np.float32)
    sn = 2.0
    out["cos_c"] = ((out["cos_n"] + out["cos_a"]) / np.sqrt(qn[q] * sn)).astype(np.float32)
    for v in VIEWS:
        out[f"rank_{v}"] = ranks[v].to_numpy()
    return out


def block_country(split, country, K, N, max_df, retr_max_df, chunk, n_threads, mt, bench=0):
    s1 = load(split, "S1", country)
    views = {v: View(v, s1, max_df, retr_max_df) for v in VIEWS}
    s1_ids = s1["i"].to_numpy()
    del s1
    mt.stage(f"{split} {country}: fit S1 ({len(s1_ids):,})")
    out = []
    for src in ("S2", "S3"):
        q = load(split, src, country)
        if bench:
            q = q.sample(min(bench, len(q)), random_state=42)
        t0 = time.perf_counter()
        for a in range(0, len(q), chunk):
            qc = q.iloc[a:a + chunk]
            r = block_chunk(views, s1_ids, qc, K, N, n_threads)
            r.insert(0, "rec", qc["i"].to_numpy()[r.pop("q").to_numpy()])
            r.insert(0, "src", np.int8(int(src[1])))
            out.append(r)
            if (a // chunk) % 20 == 19:
                print(f"  {country} {src} {a + len(qc):,}/{len(q):,}", flush=True)
        n = len(q)
        del q
        mt.stage(f"{split} {country}: query {src} ({n:,}, {(time.perf_counter() - t0) / n * 1e6:.0f} us/rec)")
    pairs = pd.concat(out, ignore_index=True)
    del views, out
    gc.collect()
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--Kn", type=int, default=10)
    ap.add_argument("--Ka", type=int, default=10)
    ap.add_argument("--Ks", type=int, default=10)
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--max-df", type=float, default=0.05)
    ap.add_argument("--retr-max-df", type=int, default=5000)
    ap.add_argument("--chunk", type=int, default=20_000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--countries", nargs="*")
    ap.add_argument("--bench", type=int, default=0, help="query sample size per source (0 = full)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    K = {"n": a.Kn, "a": a.Ka, "s": a.Ks}
    mt = MemTracker()
    d = WORK / ("blocks2_bench" if a.bench else "blocks2")
    d.mkdir(parents=True, exist_ok=True)
    for c in a.countries or countries(a.split):
        pairs = block_country(a.split, c, K, a.N, a.max_df, a.retr_max_df, a.chunk, a.threads, mt, a.bench)
        pairs.to_parquet(d / f"{a.split}_{c}{a.tag}.parquet", index=False)
        print(f"{a.split} {c}: {len(pairs):,} pairs, "
              f"{len(pairs) / pairs[['src', 'rec']].drop_duplicates().shape[0]:.1f} per record")
        del pairs
        gc.collect()


if __name__ == "__main__":
    main()
