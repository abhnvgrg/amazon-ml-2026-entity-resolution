"""Blocking v1: per country, the top-K S1 entities for every S2/S3 record (D-006, D-008).

Each country gets its own TF-IDF space fit on its S1 records only: char 3-grams of the name
core (legal suffixes removed) plus word tokens of the address, features with S1 doc freq
above max_df dropped, each view L2-normalized, hstacked and normalized again (cosine).
Search is two-stage: top-N by rare features (doc freq <= retr_max_df) with sparse_dot_topn,
then rescored with the full vectors to keep the top-K.

Output: work/blocks/{split}_{country}_K{K}.parquet (src, rec, s1, rank, cos).
Run: python src/blocking.py --split train|test --K 10
"""

from __future__ import annotations

import argparse
import gc

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

from common import WORK, MemTracker

TOKEN = r"[\ẁ-ͯऀ-෿]+"


def countries(split: str) -> list[str]:
    c = pq.read_table(WORK / "norm" / f"{split}_S1.parquet", columns=["country"]).column(0)
    return sorted(set(c.to_pylist()))


def load(split: str, src: str, country: str, cols=("i", "core", "addr")) -> pd.DataFrame:
    return pd.read_parquet(WORK / "norm" / f"{split}_{src}.parquet", columns=list(cols),
                           filters=[("country", "==", country)])


class Featurizer:
    def __init__(self, w_name: float = 1.0, max_df: float = 0.05):
        self.w = w_name
        kw = dict(max_df=max_df, dtype=np.float32, sublinear_tf=True)
        self.nv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), **kw)
        self.av = TfidfVectorizer(token_pattern=TOKEN, lowercase=False, **kw)

    def fit_transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        return self._combine(self.nv.fit_transform(df["core"].tolist()),
                             self.av.fit_transform(df["addr"].tolist()))

    def transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        return self._combine(self.nv.transform(df["core"].tolist()),
                             self.av.transform(df["addr"].tolist()))

    def _combine(self, xn, xa):
        x = sp.hstack([self.w * normalize(xn), normalize(xa)], format="csr", dtype=np.float32)
        return normalize(x)


def topk_rows(rows: np.ndarray, score: np.ndarray, K: int):
    o = np.lexsort((-score, rows))
    r = rows[o]
    start = np.r_[0, np.flatnonzero(np.diff(r)) + 1]
    rank = np.arange(len(r)) - np.repeat(start, np.diff(np.r_[start, len(r)]))
    k = rank < K
    return o[k], rank[k]


def block_country(split: str, country: str, K: int, N: int, w_name: float, max_df: float,
                  retr_max_df: int, chunk: int, n_threads: int, mt: MemTracker) -> pd.DataFrame:
    s1 = load(split, "S1", country)
    full, retr = Featurizer(w_name, max_df), Featurizer(w_name, retr_max_df)
    x1 = full.fit_transform(s1)
    x1r_t = retr.fit_transform(s1).T.tocsr()
    s1_ids = s1["i"].to_numpy()
    del s1
    mt.stage(f"{split} {country}: fit S1 ({len(s1_ids):,}, nnz full {x1.nnz:,} retr {x1r_t.nnz:,})")
    out = []
    for src in ("S2", "S3"):
        q = load(split, src, country)
        for a in range(0, len(q), chunk):
            qc = q.iloc[a:a + chunk]
            r = sp_matmul_topn(retr.transform(qc), x1r_t, top_n=N, n_threads=n_threads).tocsr()
            rows = np.repeat(np.arange(len(qc)), np.diff(r.indptr))
            cols = r.indices
            cos = np.asarray(full.transform(qc)[rows].multiply(x1[cols]).sum(axis=1)).ravel()
            keep, rank = topk_rows(rows, cos, K)
            out.append(pd.DataFrame({
                "src": np.int8(int(src[1])), "rec": qc["i"].to_numpy()[rows[keep]],
                "s1": s1_ids[cols[keep]], "rank": rank.astype(np.int8),
                "cos": cos[keep].astype(np.float32)}))
            if (a // chunk) % 20 == 19:
                print(f"  {country} {src} {a + len(qc):,}/{len(q):,}", flush=True)
        n = len(q)
        del q
        mt.stage(f"{split} {country}: query {src} ({n:,})")
    pairs = pd.concat(out, ignore_index=True)
    del x1, x1r_t, full, retr, out
    gc.collect()
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--w-name", type=float, default=1.0)
    ap.add_argument("--max-df", type=float, default=0.05)
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--retr-max-df", type=int, default=5000)
    ap.add_argument("--chunk", type=int, default=20_000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--countries", nargs="*")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    mt = MemTracker()
    (WORK / "blocks").mkdir(parents=True, exist_ok=True)
    for c in a.countries or countries(a.split):
        pairs = block_country(a.split, c, a.K, a.N, a.w_name, a.max_df, a.retr_max_df,
                              a.chunk, a.threads, mt)
        pairs.to_parquet(WORK / "blocks" / f"{a.split}_{c}_K{a.K}{a.tag}.parquet", index=False)
        print(f"{a.split} {c}: {len(pairs):,} pairs")
        del pairs
        gc.collect()


if __name__ == "__main__":
    main()
