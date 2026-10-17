"""Pairwise matcher features (60 in the final model), computed per country.

Candidates come from blocking v2 (env ER_BLOCKS=v3, the default; v1 reproduces sub-03) and are
pruned with the rule chosen in eval_blocking3.py. Feature groups: retrieval rank / gap / tie
and source ranks; name-only vs address-only cosines; name similarities (rapidfuzz, legal-suffix
and extra-token deltas, transliteration skeleton); address-number agreement and drift; graph
and sibling context from PREDICTED rank-0 assignments (never ground truth); lengths and source.
Country is used only to partition work, never as a feature.

Off by default: ER_KN=5 adds the sub-06 name-only candidates, ER_KD the sub-07 dense ones.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from blocking import TOKEN
from common import WORK
from normalize import LEGAL
from translit import has_indic, skeleton

TOPK, COS_FLOOR, NNUM = 5, 0.3, 6
BLOCKS = os.environ.get("ER_BLOCKS", "v3")
V3_DELTA, V3_K, V3_KS, V3_KA = 0.1, 10, 5, 0
V3_KN = int(os.environ.get("ER_KN", "0"))
V3_KD = int(os.environ.get("ER_KD", "0"))
SRC_COLS = (["rank_v1", "rank_a", "rank_s"] + (["rank_n"] if V3_KN > 0 else []) + (["rank_d"] if V3_KD > 0 else [])) if BLOCKS == "v3" else []
DENSE_COLS = ["cos_d"] if BLOCKS == "v3" and V3_KD > 0 else []
NUM_RE = re.compile(r"\d+")
SIB_RANK = 2


def rec_key(src, rec) -> np.ndarray:
    return np.asarray(src, np.int64) * 10**12 + np.asarray(rec, np.int64)


def dense_cos(split: str, country: str, key: np.ndarray, s1: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    emb = WORK / "embeddings"
    rk = np.load(emb / f"{split}_{country}_R_keys.npy")
    sk = np.load(emb / f"{split}_{country}_S1_keys.npy")
    R = np.load(emb / f"{split}_{country}_R.npy", mmap_mode="r")
    S = np.load(emb / f"{split}_{country}_S1.npy")
    ri, si = np.searchsorted(rk, key), np.searchsorted(sk, s1)
    assert (rk[ri] == key).all() and (sk[si] == s1).all()
    out = np.empty(len(key), np.float32)
    for lo in range(0, len(key), chunk):
        a = R[ri[lo:lo + chunk]].astype(np.float32)
        out[lo:lo + chunk] = np.einsum("ij,ij->i", a, S[si[lo:lo + chunk]].astype(np.float32))
    return out


def _nums(addr: str) -> list[int]:
    out = []
    for t in NUM_RE.findall(addr):
        v = int(t[-15:])
        if v not in out:
            out.append(v)
    return out


_LEGAL_CODES: dict[str, int] = {"": 0}


def _nonnum(a: str) -> str:
    return " ".join(t for t in a.split() if not t.isdigit())


class Entities:

    def __init__(self, df: pd.DataFrame, n: int | None = None, pos: np.ndarray | None = None):
        m = len(df)
        n = m if n is None else n
        pos = np.arange(m) if pos is None else pos
        self.loaded = np.zeros(n, bool)
        self.loaded[pos] = True
        self.name = np.full(n, "", dtype=object)
        self.core = np.full(n, "", dtype=object)
        self.addr = np.full(n, "", dtype=object)
        name, core, addr = df["name"].tolist(), df["core"].tolist(), df["addr"].tolist()
        self.name[pos], self.core[pos], self.addr[pos] = name, core, addr
        self.indic = np.zeros(n, bool)
        self.indic[pos] = np.fromiter((has_indic(x) for x in name), bool, m)
        self.legal = np.zeros(n, np.int32)
        self.legal[pos] = np.fromiter((_LEGAL_CODES.setdefault(" ".join(sorted({t for t in x.split() if t in LEGAL})),
                                                                len(_LEGAL_CODES)) for x in name), np.int32, m)
        nums = np.full((m, NNUM), -1, np.int64)
        n_num = np.zeros(m, np.int16)
        for i, a in enumerate(addr):
            v = _nums(a)
            n_num[i] = len(v)
            v = v[:NNUM]
            nums[i, :len(v)] = v
        self.nums = np.full((n, NNUM), -1, np.int64)
        self.nums[pos] = nums
        self.n_num = np.zeros(n, np.int16)
        self.n_num[pos] = n_num
        self.len_core = np.zeros(n, np.int16)
        self.len_core[pos] = np.fromiter((len(x) for x in core), np.int16, m)
        self.ntok_core = np.zeros(n, np.int16)
        self.ntok_core[pos] = np.fromiter((x.count(" ") + 1 if x else 0 for x in core), np.int16, m)
        self.len_addr = np.zeros(n, np.int16)
        self.len_addr[pos] = np.fromiter((len(x) for x in addr), np.int16, m)


GRAPH_COLS = ["rank", "n_cand", "cos_top", "gap_top", "gap_next", "gap_12", "tie_12", "tie",
              "indeg_s2", "indeg_s3", "indeg", "sib_other_src", "sib_rank", "cos_vs_s1max"] + SRC_COLS + DENSE_COLS
SIB_COLS = ["sib_max_name", "sib_max_addr"]
BASE_COLS = ["src", "rec", "s1", "cos"]


class Country:

    def __init__(self, split: str, country: str):
        self.split, self.country = split, country
        cols = ["i", "name", "core", "addr"]
        flt = [("country", "==", country)]
        s1 = pd.read_parquet(WORK / "norm" / f"{split}_S1.parquet", columns=cols,
                             filters=flt).sort_values("i", ignore_index=True)
        self.S = Entities(s1)
        self.S.ids = s1["i"].to_numpy()
        keys = [rec_key(np.int8(k), pq.read_table(WORK / "norm" / f"{split}_S{k}.parquet", columns=["i"],
                                                  filters=flt).column("i").to_numpy()) for k in (2, 3)]
        self.r_key = np.sort(np.concatenate(keys))
        self.R: Entities | None = None
        kw = dict(max_df=0.05, dtype=np.float32, sublinear_tf=True)
        nv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), **kw)
        av = TfidfVectorizer(token_pattern=TOKEN, lowercase=False, **kw)
        self.sn = normalize(nv.fit_transform(self.S.core.tolist()))
        self.sa = normalize(av.fit_transform(self.S.addr.tolist()))
        self.nv, self.av = nv, av

    def load_records(self, pos: np.ndarray) -> None:
        self.R = None
        want = np.zeros(len(self.r_key), bool)
        want[pos] = True
        parts, ppos = [], []
        for k in (2, 3):
            t = pq.read_table(WORK / "norm" / f"{self.split}_S{k}.parquet", columns=["i", "name", "core", "addr"],
                              filters=[("country", "==", self.country)])
            p = np.searchsorted(self.r_key, rec_key(np.int8(k), t.column("i").to_numpy()))
            m = want[p]
            parts.append(t.filter(pa.array(m)).to_pandas())
            ppos.append(p[m])
            del t
        self.R = Entities(pd.concat(parts, ignore_index=True), len(self.r_key), np.concatenate(ppos))

    def needed_records(self, c: pd.DataFrame) -> np.ndarray:
        sib_s = np.unique(c["s"].to_numpy()[c["rank"].to_numpy() < SIB_RANK])
        top = self._top
        sib_r = top["r"].to_numpy()[np.isin(top["s"].to_numpy(), sib_s)]
        return np.union1d(c["r"].to_numpy(), sib_r)

    def save_graph(self, path) -> None:
        t = self._top
        pd.DataFrame({"key": self.r_key[t["r"].to_numpy()], "s1": self.S.ids[t["s"].to_numpy()],
                      "cos": t["cos"].to_numpy()}).to_parquet(path, index=False)

    def candidates(self, remove_s1: np.ndarray | None = None, K: int = 10) -> pd.DataFrame:
        if BLOCKS == "v3":
            t = pq.read_table(WORK / "blocks3" / f"{self.split}_{self.country}.parquet")
            srk = np.stack([t.column(x).to_numpy() for x in SRC_COLS], axis=1)
        else:
            t = pq.read_table(WORK / "blocks" / f"{self.split}_{self.country}_K{K}.parquet",
                              columns=["src", "rec", "s1", "cos"])
            srk = np.zeros((t.num_rows, 0), np.int8)
        src, s1 = t.column("src").to_numpy(), t.column("s1").to_numpy()
        key = rec_key(src, t.column("rec").to_numpy())
        cos = t.column("cos").to_numpy().astype(np.float32)
        del t
        if remove_s1 is not None:
            m = ~np.isin(s1, remove_s1)
            src, s1, key, cos, srk = src[m], s1[m], key[m], cos[m], srk[m]
            del m
        brk = np.diff(key) != 0
        st = np.r_[0, np.flatnonzero(brk) + 1]
        grouped = len(np.unique(key[st])) == len(st) and bool(((np.diff(cos) <= 0) | brk).all())
        del brk, st
        if not grouped:
            o = np.lexsort((-cos, key))
            key, cos, s1, src, srk = key[o], cos[o], s1[o], src[o], srk[o]
            del o
        n = len(key)
        start = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
        cnt = np.diff(np.r_[start, n]).astype(np.int32)
        grp = np.repeat(np.arange(len(start), dtype=np.int32), cnt)
        rank = (np.arange(n, dtype=np.int32) - start[grp]).astype(np.int32)
        is0 = rank == 0
        ns = len(self.S.ids)
        t_s = np.searchsorted(self.S.ids, s1[is0])
        t_r = np.searchsorted(self.r_key, key[is0])
        t_cos, t_src = cos[is0], src[is0]
        self._top = pd.DataFrame({"s": t_s, "r": t_r, "cos": t_cos})
        c2 = np.bincount(t_s[t_src == 2], minlength=ns)
        c3 = np.bincount(t_s[t_src == 3], minlength=ns)
        cmax = np.zeros(ns, np.float32)
        np.maximum.at(cmax, t_s, t_cos)
        ts = np.lexsort((t_cos, t_s))
        tkey = t_s[ts] * 4.0 + t_cos[ts]
        del ts, t_s, t_r, t_cos, t_src
        last = np.r_[start[1:] - 1, n - 1]
        nxt = np.r_[cos[1:], np.float32(0)]
        nxt[last] = 0
        prv = np.r_[np.float32(np.inf), cos[:-1]]
        prv[start] = np.inf
        cos0 = cos[start]
        cos1 = np.where(cnt > 1, cos[np.minimum(start + 1, n - 1)], 0).astype(np.float32)
        if BLOCKS == "v3":
            keep = ((rank < V3_K) & (is0 | (cos0[grp] - cos <= V3_DELTA))) | (srk[:, 2] < V3_KS) | (srk[:, 1] < V3_KA)
            if V3_KN > 0:
                keep |= srk[:, SRC_COLS.index("rank_n")] < V3_KN
            if V3_KD > 0:
                keep |= srk[:, SRC_COLS.index("rank_d")] < V3_KD
        else:
            keep = (rank < TOPK) & (is0 | (cos >= COS_FLOOR))
        g = grp[keep]
        k_srk = srk[keep]
        k_key, k_cos, k_s1, k_src = key[keep], cos[keep], s1[keep], src[keep]
        k_rank, k_is0 = rank[keep], is0[keep]
        k_nxt, k_prv = nxt[keep], prv[keep]
        del grp, key, cos, s1, src, rank, is0, nxt, prv, keep, srk
        s = np.searchsorted(self.S.ids, k_s1)
        r = np.searchsorted(self.r_key, k_key)
        assert (self.r_key[r] == k_key).all() and (self.S.ids[s] == k_s1).all()
        c = pd.DataFrame({"src": k_src, "rec": k_key % 10**12, "s1": k_s1, "cos": k_cos,
                          "key": k_key, "r": r, "s": s})
        c["rank"] = k_rank.astype(np.int8)
        c["n_cand"] = np.minimum(cnt[g], 127).astype(np.int8)
        c["cos_top"] = cos0[g]
        c["gap_top"] = cos0[g] - k_cos
        c["gap_next"] = k_cos - k_nxt
        c["gap_12"] = cos0[g] - cos1[g]
        c["tie_12"] = (cos0[g] == cos1[g]) & (cnt[g] > 1)
        c["tie"] = (k_cos == k_nxt) | (k_cos == k_prv)
        c["indeg_s2"] = (c2[s] - (k_is0 & (k_src == 2))).astype(np.int16)
        c["indeg_s3"] = (c3[s] - (k_is0 & (k_src == 3))).astype(np.int16)
        c["indeg"] = c["indeg_s2"] + c["indeg_s3"]
        c["sib_other_src"] = np.where(k_src == 2, c3[s], c2[s]) > 0
        end = np.searchsorted(tkey, s * 4.0 + 3.0)
        c["sib_rank"] = (end - np.searchsorted(tkey, s * 4.0 + k_cos, side="right")).astype(np.int16)
        c["cos_vs_s1max"] = k_cos - np.where(k_is0, np.maximum(cmax[s], k_cos), cmax[s])
        for j, x in enumerate(SRC_COLS):
            c[x] = k_srk[:, j]
        if DENSE_COLS:
            c["cos_d"] = dense_cos(self.split, self.country, c["key"].to_numpy(), c["s1"].to_numpy())
        return c

    def features(self, c: pd.DataFrame, chunk: int = 2_000_000, reuse: pd.DataFrame | None = None):
        self.load_records(self.needed_records(c))
        for i in range(0, len(c), chunk):
            yield self._features(c.iloc[i:i + chunk].reset_index(drop=True), reuse)
        self.R = None

    def _features(self, c: pd.DataFrame, reuse: pd.DataFrame | None = None) -> pd.DataFrame:
        f = c[BASE_COLS + GRAPH_COLS].copy()
        f["src"] = f["src"].astype(np.int8)
        if reuse is None:
            pf = self._pair_feats(c)
        else:
            pf = c[["key", "s1"]].merge(reuse, on=["key", "s1"], how="left").drop(columns=["key", "s1"])
            miss = pf.iloc[:, 0].isna().to_numpy()
            pf = pf.astype(np.float32)
            if miss.any():
                new = self._pair_feats(c[miss].reset_index(drop=True)).astype(np.float32)
                pf.loc[miss, new.columns] = new.to_numpy()
        f = pd.concat([f, pf.reset_index(drop=True)], axis=1)
        f = f.assign(**self._sibling_sims(c))
        for col in f.columns:
            if f[col].dtype == np.float64:
                f[col] = f[col].astype(np.float32)
        return f

    def _pair_feats(self, c: pd.DataFrame) -> pd.DataFrame:
        r, s = c["r"].to_numpy(), c["s"].to_numpy()
        R, S = self.R, self.S
        assert R.loaded[r].all(), "record texts not loaded"
        f = pd.DataFrame(index=range(len(c)))
        ur, ri = np.unique(r, return_inverse=True)
        us, si = np.unique(s, return_inverse=True)
        f["cos_n"] = _rowdot(normalize(self.nv.transform(R.core[ur].tolist())), self.sn, ri, s)
        f["cos_a"] = _rowdot(normalize(self.av.transform(R.addr[ur].tolist())), self.sa, ri, s)
        f["cos_na_diff"] = f["cos_n"] - f["cos_a"]
        f["cos_na_prod"] = f["cos_n"] * f["cos_a"]
        f["name_hi_addr_lo"] = (f["cos_n"] >= 0.8) & (f["cos_a"] < 0.5)
        f["addr_hi_name_lo"] = (f["cos_a"] >= 0.8) & (f["cos_n"] < 0.5)
        a, b = R.core[r], S.core[s]
        f["n_tset"] = cpdist(a, b, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        f["n_tsort"] = cpdist(a, b, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float32)
        f["n_partial"] = cpdist(a, b, scorer=fuzz.partial_ratio, workers=-1, dtype=np.float32)
        f["n_jw"] = cpdist(a, b, scorer=JaroWinkler.normalized_similarity, workers=-1, dtype=np.float32)
        sk_r = np.array([skeleton(x) for x in R.core[ur]], dtype=object)[ri]
        sk_s = np.array([skeleton(x) for x in S.core[us]], dtype=object)[si]
        f["n_skel"] = cpdist(sk_r, sk_s, scorer=fuzz.ratio, workers=-1, dtype=np.float32)
        f["n_full_ratio"] = cpdist(R.name[r], S.name[s], scorer=fuzz.ratio, workers=-1, dtype=np.float32)
        f["name_eq"] = R.name[r] == S.name[s]
        f["core_eq"] = a == b
        extra_r, extra_s = _extra_tokens(a, b)
        f["n_extra_rec"], f["n_extra_s1"] = extra_r, extra_s
        lr, ls = R.legal[r], S.legal[s]
        f["legal_eq"] = lr == ls
        f["legal_rec"], f["legal_s1"] = lr != 0, ls != 0
        f["indic"] = R.indic[r]
        f["rec_addr_empty"] = R.addr[r] == ""
        f["a_tset"] = cpdist(R.addr[r], S.addr[s], scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        nn_r = np.array([_nonnum(x) for x in R.addr[ur]], dtype=object)[ri]
        nn_s = np.array([_nonnum(x) for x in S.addr[us]], dtype=object)[si]
        f["a_nonnum_tset"] = cpdist(nn_r, nn_s, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        f = f.assign(**_num_feats(R.nums[r], S.nums[s], R.n_num[r], S.n_num[s]))
        f["len_core_rec"], f["len_core_s1"] = R.len_core[r], S.len_core[s]
        f["ntok_rec"], f["ntok_s1"] = R.ntok_core[r], S.ntok_core[s]
        f["len_addr_rec"], f["len_addr_s1"] = R.len_addr[r], S.len_addr[s]
        return f

    def _sibling_sims(self, c: pd.DataFrame) -> dict:
        n = len(c)
        out = {k: np.full(n, np.nan, np.float32) for k in ("sib_max_name", "sib_max_addr")}
        idx = np.flatnonzero(c["rank"].to_numpy() < SIB_RANK)
        top = self._top[["s", "r"]]
        q = pd.DataFrame({"pid": idx, "s": c["s"].to_numpy()[idx], "r_self": c["r"].to_numpy()[idx]})
        j = q.merge(top, on="s")
        j = j[j["r"] != j["r_self"]]
        if len(j):
            R = self.R
            assert R.loaded[j["r"].to_numpy()].all(), "sibling texts not loaded"
            j["nm"] = cpdist(R.core[j["r_self"].to_numpy()], R.core[j["r"].to_numpy()],
                             scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
            j["ad"] = cpdist(R.addr[j["r_self"].to_numpy()], R.addr[j["r"].to_numpy()],
                             scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
            g = j.groupby("pid")[["nm", "ad"]].max()
            out["sib_max_name"][g.index.to_numpy()] = g["nm"].to_numpy()
            out["sib_max_addr"][g.index.to_numpy()] = g["ad"].to_numpy()
        return out


def _rowdot(a, b, ra, rb, step=2_000_000) -> np.ndarray:
    out = np.empty(len(ra), np.float32)
    for i in range(0, len(ra), step):
        out[i:i + step] = np.asarray(a[ra[i:i + step]].multiply(b[rb[i:i + step]]).sum(axis=1)).ravel()
    return out


def _extra_tokens(a: np.ndarray, b: np.ndarray):
    er = np.empty(len(a), np.int8)
    es = np.empty(len(a), np.int8)
    for i, (x, y) in enumerate(zip(a, b)):
        sx, sy = set(x.split()), set(y.split())
        er[i] = min(len(sx - sy), 127)
        es[i] = min(len(sy - sx), 127)
    return er, es


def _num_feats(nr: np.ndarray, ns: np.ndarray, cr: np.ndarray, cs: np.ndarray) -> dict:
    vr, vs = nr >= 0, ns >= 0
    eq = (nr[:, :, None] == ns[:, None, :]) & vr[:, :, None] & vs[:, None, :]
    r_in_s = eq.any(axis=2)
    s_in_r = eq.any(axis=1)
    n_r_only = (vr & ~r_in_s).sum(1)
    n_s_only = (vs & ~s_in_r).sum(1)
    shared = (vs & s_in_r).sum(1)
    both = (cr > 0) & (cs > 0)
    ur, us = vr & ~r_in_s, vs & ~s_in_r
    d = np.abs(nr[:, :, None] - ns[:, None, :]).astype(np.float64)
    d[~(ur[:, :, None] & us[:, None, :])] = np.inf
    dmin = d.min(axis=(1, 2))
    first_eq = np.where(both, nr[:, 0] == ns[:, 0], False)
    first_diff = np.where(both, np.abs(nr[:, 0] - ns[:, 0]).astype(np.float64), np.nan)
    return {
        "num_n_rec": cr.astype(np.int8), "num_n_s1": cs.astype(np.int8),
        "num_first_eq": first_eq,
        "num_first_logdiff": np.log1p(first_diff).astype(np.float32),
        "num_all_eq": both & (n_r_only == 0) & (n_s_only == 0),
        "num_s1_subset": (cs > 0) & (n_s_only == 0),
        "num_shared": shared.astype(np.int8),
        "num_rec_only": n_r_only.astype(np.int8), "num_s1_only": n_s_only.astype(np.int8),
        "num_min_logdiff": np.where(np.isinf(dmin), np.nan, np.log1p(dmin)).astype(np.float32),
    }
