"""Diagnostic (D-014): which name representation recovers Indic-script records missed by blocking?

Compares a consonant skeleton, the learned token dictionary (name-only, and name plus address)
and plain transliteration on sampled missed val pairs. Dictionary name + address won (0.94).
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from blocking import TOKEN
from common import SEED, WORK, MemTracker, is_val
from errors import rec_info
from features import rec_key
from matcher import FEATS
from normalize import LEGAL
from recovery_diag import existing, topn_pairs
from translit import has_indic, skeleton, translit

C, NQ, K, MIN_N, MIN_SHARE = "India", 5000, 5, 3, 0.7


def names(split: str, k: str, cols=("i", "name", "core", "addr")) -> pd.DataFrame:
    return ds.dataset(WORK / "norm" / f"{split}_{k}.parquet").to_table(
        columns=list(cols), filter=ds.field("country") == C).to_pandas()


def recs(split: str) -> pd.DataFrame:
    out = []
    for k in (2, 3):
        d = names(split, f"S{k}")
        d["key"] = rec_key(np.int8(k), d.pop("i").to_numpy())
        out.append(d)
    d = pd.concat(out, ignore_index=True)
    d["indic"] = [has_indic(x) for x in d["name"]]
    return d.set_index("key")


def learn_dict(gt: pd.DataFrame, r: pd.DataFrame, s1: pd.DataFrame) -> dict[str, str]:
    g = gt[~is_val(gt["s1"].to_numpy())]
    rn = r["name"].reindex(g["key"]).to_numpy()
    sn = s1["name"].reindex(g["s1"]).to_numpy()
    cnt: dict[str, Counter] = defaultdict(Counter)
    n_used = 0
    for a, b in zip(rn, sn):
        if not isinstance(a, str) or not isinstance(b, str) or not has_indic(a) or has_indic(b):
            continue
        ta, tb = a.split(), b.split()
        if len(ta) != len(tb):
            continue
        n_used += 1
        for x, y in zip(ta, tb):
            if has_indic(x):
                cnt[x][y] += 1
    d = {}
    for x, c in cnt.items():
        n = sum(c.values())
        y, m = c.most_common(1)[0]
        if n >= MIN_N and m / n >= MIN_SHARE:
            d[x] = y
    print(f"dictionary: {n_used:,} aligned train pairs, {len(cnt):,} Indic tokens seen, "
          f"{len(d):,} kept (count >= {MIN_N}, share >= {MIN_SHARE})")
    return d


def coverage(d: dict, names_: pd.Series, label: str) -> None:
    tot = cov = 0
    types, ctypes = set(), set()
    for s in names_:
        for t in s.split():
            if has_indic(t):
                tot += 1
                types.add(t)
                if t in d:
                    cov += 1
                    ctypes.add(t)
    print(f"  {label}: Indic token occurrences covered {cov / max(tot, 1):.4f} ({cov:,}/{tot:,}); "
          f"types {len(ctypes) / max(len(types), 1):.4f}")


def map_name(s: str, d: dict) -> str:
    out = []
    for t in s.split():
        m = d.get(t) or translit(t)
        if m not in LEGAL:
            out.append(m)
    return " ".join(out) or translit(s)


def main() -> None:
    mt = MemTracker()
    rng = np.random.default_rng(SEED)
    s1 = names("train", "S1").set_index("i")
    r = recs("train")
    gt = pd.read_parquet(WORK / "norm" / "train_gt.parquet")
    gt = gt[np.isin(gt["s1"].to_numpy(), s1.index.to_numpy())].reset_index(drop=True)
    gt["key"] = rec_key(gt["src"].to_numpy().astype(np.int8), gt["rec"].to_numpy())
    mt.stage("loaded")

    vs = pd.read_parquet(FEATS / f"val_sample_{C}.parquet")["s1"].to_numpy()
    lab = pd.read_parquet(FEATS / f"val_{C}.parquet", columns=["key", "label"])
    g = gt[np.isin(gt["s1"].to_numpy(), vs)]
    miss = g[~np.isin(g["key"].to_numpy(), lab.loc[lab["label"], "key"].to_numpy())]
    miss = miss[r["indic"].reindex(miss["key"]).fillna(False).to_numpy()]
    print(f"missed Indic-script pairs: {len(miss):,}; sampling {NQ:,}")
    miss = miss.iloc[rng.choice(len(miss), min(NQ, len(miss)), replace=False)].reset_index(drop=True)
    vrec = lab["key"].unique()
    vrec = vrec[r["indic"].reindex(vrec).fillna(False).to_numpy()]
    samp = rng.choice(vrec, NQ, replace=False)
    del lab
    ex = existing(C)
    exs = ex[np.isin(ex["key"].to_numpy(), samp)][["key", "s1"]]
    del ex
    qkeys = np.unique(np.r_[miss["key"].to_numpy(), samp])
    qname = r["name"].reindex(qkeys).to_numpy()
    qcore = r["core"].reindex(qkeys).to_numpy()
    s1_ids = s1.index.to_numpy()

    t0 = time.perf_counter()
    d = learn_dict(gt, r, s1)
    t_dict = time.perf_counter() - t0
    print("coverage of Indic tokens by the dictionary:")
    vkeys = gt.loc[is_val(gt["s1"].to_numpy()), "key"].to_numpy()
    vn = r["name"].reindex(vkeys)
    coverage(d, vn[r["indic"].reindex(vkeys).fillna(False).to_numpy()].dropna(), "val-split matched records")
    tr = recs("test")
    coverage(d, tr.loc[tr["indic"], "name"], f"TEST India records ({int(tr['indic'].sum()):,} Indic)")
    n_test_indic = int(tr["indic"].sum())
    del tr, r
    mt.stage("dictionary + coverage")

    kw = dict(max_df=0.05, dtype=np.float32, sublinear_tf=True)
    t0 = time.perf_counter()
    sv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), **kw)
    xs_sk = normalize(sv.fit_transform([skeleton(x) for x in s1["core"]]))
    t_sk_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    nv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), **kw)
    xs_n = normalize(nv.fit_transform(s1["core"].tolist()))
    t_n_fit = time.perf_counter() - t0
    av = TfidfVectorizer(token_pattern=TOKEN, lowercase=False, **kw)
    xs_c = normalize(sp.hstack([xs_n, normalize(av.fit_transform(s1["addr"].tolist()))], format="csr"))
    qaddr = rec_info(qkeys, cols=("addr",)).reindex(qkeys)["addr"].fillna("").tolist()
    comb = lambda names_: normalize(sp.hstack([normalize(nv.transform(names_)), normalize(av.transform(qaddr))], format="csr"))
    mt.stage("S1 vectors")

    mapped = [map_name(x, d) for x in qname]
    queries = {
        "(i) skeleton": (lambda: sv.transform([skeleton(x) for x in qcore]), xs_sk, t_sk_fit),
        "(iii-lat) dict, Latin char-3": (lambda: nv.transform(mapped), xs_n, t_n_fit + t_dict),
        "(iii-sk) dict, skeleton": (lambda: sv.transform([skeleton(x) for x in mapped]), xs_sk, t_sk_fit + t_dict),
        "(iii-comb) dict name + address (v1 combined)": (lambda: comb(mapped), xs_c, t_n_fit + t_dict),
        "(ref) v1 combined, raw Indic name + address": (lambda: comb(list(qcore)), xs_c, t_n_fit),
        "(ref) translit char-3 [= recovery_diag e]": (lambda: nv.transform([translit(x) for x in qcore]), xs_n, t_n_fit),
    }
    rows, hits = [], {}
    for name, (qf, X, t_fit) in queries.items():
        t0 = time.perf_counter()
        rr, cc, _ = topn_pairs(normalize(qf()), X.T.tocsr(), K)
        t_q = time.perf_counter() - t0
        f = pd.DataFrame({"key": qkeys[rr], "s1": s1_ids[cc]}).drop_duplicates()
        h = miss.merge(f, on=["key", "s1"], how="left", indicator=True)["_merge"].eq("both").to_numpy()
        hits[name] = h
        fs = f[np.isin(f["key"].to_numpy(), samp)].merge(exs, on=["key", "s1"], how="left", indicator=True)
        new = (fs["_merge"] == "left_only").sum() / len(samp)
        rows.append({"method": name, "recovered": h.mean(), "new_pairs_per_indic_rec": new,
                     "fit_s": t_fit, "search_s_per_1k": t_q / len(qkeys) * 1000,
                     "est_search_min_test_indic": t_q / len(qkeys) * n_test_indic / 60})
    rows.append({"method": "union (iii-lat)+(iii-comb)",
                 "recovered": (hits["(iii-lat) dict, Latin char-3"] | hits["(iii-comb) dict name + address (v1 combined)"]).mean()})
    rows.append({"method": "union (i)+(iii-lat)+(iii-sk)",
                 "recovered": (hits["(i) skeleton"] | hits["(iii-lat) dict, Latin char-3"]
                               | hits["(iii-sk) dict, skeleton"]).mean()})
    pd.set_option("display.width", 220)
    print(f"\nRecovery on {len(miss):,} missed Indic-script pairs (true S1 in top-{K}); cost on "
          f"{len(samp):,} random Indic val records; test India Indic records {n_test_indic:,}:")
    print(pd.DataFrame(rows).set_index("method").round(4).to_string())
    print("\nexamples (record name -> mapped):")
    for x in qname[:8]:
        print(f"  {x}  ->  {map_name(x, d)}  |  sk: {skeleton(map_name(x, d))}")
    mt.stage("done")


if __name__ == "__main__":
    main()
