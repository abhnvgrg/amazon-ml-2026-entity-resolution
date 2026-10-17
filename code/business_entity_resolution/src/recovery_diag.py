"""Diagnostic: which extra retrievals would recover val pairs missed by the sub-03 candidates?

Compares exact name-only, address-only, S1-side, sibling and transliterated-name searches by
recovered pairs and by extra candidates per record. Its findings led to blocking v2.
"""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

from blocking import TOKEN
from common import SEED, WORK, MemTracker
from errors import rec_info, s1_info
from features import rec_key
from matcher import FEATS, val_entities
from translit import has_indic, translit

KQ, KS, SIB_T, N_REC, N_S1, TH = 5, 10, 0.9, 20_000, 10_000, 8
METHODS = ["a_name", "b_addr", "c_s1side", "d_sibling", "e_translit"]


def topn_pairs(A, BT, n, threshold=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R, C, V = [], [], []
    for lo in range(0, A.shape[0], 5000):
        r = sp_matmul_topn(A[lo:lo + 5000], BT, top_n=n, threshold=threshold, n_threads=TH).tocsr()
        R.append(lo + np.repeat(np.arange(r.shape[0]), np.diff(r.indptr)))
        C.append(r.indices.astype(np.int64))
        V.append(r.data)
    return np.concatenate(R), np.concatenate(C), np.concatenate(V)


def load_recs(c: str) -> pd.DataFrame:
    out = []
    for k in (2, 3):
        t = ds.dataset(WORK / "norm" / f"train_S{k}.parquet").to_table(
            columns=["i", "core", "addr"], filter=ds.field("country") == c).to_pandas()
        t["key"] = rec_key(np.int8(k), t.pop("i").to_numpy())
        out.append(t)
    return pd.concat(out, ignore_index=True).sort_values("key", ignore_index=True)


def existing(c: str) -> pd.DataFrame:
    b = pd.read_parquet(WORK / "blocks" / f"train_{c}_K10.parquet")
    b["key"] = rec_key(b["src"].to_numpy(), b["rec"].to_numpy())
    b = b[(b["rank"] < 5) & ((b["rank"] == 0) | (b["cos"] >= 0.3))]
    return b[["key", "s1", "rank"]].reset_index(drop=True)


def main() -> None:
    mt = MemTracker()
    rng = np.random.default_rng(SEED)
    val_ids, val_cty, _, _ = val_entities()
    gt_all = pd.read_parquet(WORK / "norm" / "train_gt.parquet")
    best = pd.read_parquet(WORK / "best_val.parquet", columns=["key", "country"])
    res_rows, ex_rows = [], []
    for c in np.unique(val_cty):
        ents = val_ids[val_cty == c]
        gt = gt_all[np.isin(gt_all["s1"].to_numpy(), ents)].reset_index(drop=True)
        gt["key"] = rec_key(gt["src"].to_numpy().astype(np.int8), gt["rec"].to_numpy())
        lab = pd.read_parquet(FEATS / f"val_{c}.parquet", columns=["key", "label"])
        miss = gt[~np.isin(gt["key"].to_numpy(), lab.loc[lab["label"], "key"].to_numpy())].reset_index(drop=True)
        del lab
        ri = rec_info(miss["key"].to_numpy()).reindex(miss["key"])
        si = s1_info(miss["s1"].to_numpy()).reindex(miss["s1"])
        nsim = cpdist(ri["core"].fillna("").tolist(), si["core"].fillna("").tolist(),
                      scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        indic = np.array([has_indic(x) for x in ri["name"].fillna("")])
        miss["slice"] = np.select([indic, ri["addr_empty"].fillna(False).to_numpy().astype(bool), nsim < 50],
                                  ["Indic script name", "empty address", "name_sim<50 (DBA)"], "other")
        print(f"\n=== {c}: {len(miss):,} GT pairs with true S1 not in candidates", flush=True)

        s1 = ds.dataset(WORK / "norm" / "train_S1.parquet").to_table(
            columns=["i", "core", "addr"], filter=ds.field("country") == c).to_pandas()
        kw = dict(max_df=0.05, dtype=np.float32, sublinear_tf=True)
        nv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), **kw)
        av = TfidfVectorizer(token_pattern=TOKEN, lowercase=False, **kw)
        xn_s = normalize(nv.fit_transform(s1["core"].tolist()))
        xa_s = normalize(av.fit_transform(s1["addr"].tolist()))
        xc_s = normalize(sp.hstack([xn_s, xa_s], format="csr"))
        s1_ids = s1["i"].to_numpy()
        del s1
        s1_pos = pd.Series(np.arange(len(s1_ids)), index=s1_ids)

        recs = load_recs(c)
        rkeys = recs["key"].to_numpy()
        xn_r = normalize(nv.transform(recs["core"].tolist()))
        xa_r = normalize(av.transform(recs["addr"].tolist()))
        del recs
        gc.collect()
        mt.stage(f"{c}: vectors S1 {len(s1_ids):,}, records {len(rkeys):,}")

        samp = rng.choice(best.loc[best["country"] == c, "key"].to_numpy(), N_REC, replace=False)
        qkeys = np.unique(np.r_[miss["key"].to_numpy(), samp])
        qpos = np.searchsorted(rkeys, qkeys)
        assert (rkeys[qpos] == qkeys).all()
        qn, qa = xn_r[qpos], xa_r[qpos]
        qinfo = rec_info(qkeys, cols=("core",)).reindex(qkeys)
        qe = normalize(nv.transform([translit(x) for x in qinfo["core"].fillna("")]))

        found = {}

        def rec_to_s1(Q, X, n, name):
            r, col, _ = topn_pairs(Q, X.T.tocsr(), n)
            found[name] = pd.DataFrame({"key": qkeys[r], "s1": s1_ids[col]})
            mt.stage(f"{c}: {name} ({len(r):,} pairs)")

        rec_to_s1(qn, xn_s, KQ, "a_name")
        rec_to_s1(qa, xa_s, KQ, "b_addr")
        rec_to_s1(qe, xn_s, KQ, "e_translit")

        ex = existing(c)
        top1 = ex[ex["rank"] == 0].set_index("key")["s1"]
        sib = []
        for X, Q in ((xn_r, qn), (xa_r, qa)):
            r, col, _ = topn_pairs(Q, X.T.tocsr(), 50, threshold=SIB_T)
            k = rkeys[col]
            ok = k != qkeys[r]
            sib.append(pd.DataFrame({"key": qkeys[r[ok]], "sib": k[ok]}))
            gc.collect()
        sib = pd.concat(sib, ignore_index=True)
        sib["s1"] = top1.reindex(sib["sib"].to_numpy()).to_numpy()
        found["d_sibling"] = sib.dropna(subset=["s1"]).astype({"s1": np.int64})[["key", "s1"]]
        mt.stage(f"{c}: d_sibling ({len(sib):,} sibling links)")

        s1_samp = rng.choice(s1_ids, N_S1, replace=False)
        s1_q = np.unique(np.r_[miss["s1"].to_numpy(), s1_samp])
        xc_r = normalize(sp.hstack([xn_r, xa_r], format="csr"))
        del xn_r, xa_r, qn, qa, qe
        gc.collect()
        r, col, _ = topn_pairs(xc_s[s1_pos.reindex(s1_q).to_numpy()], xc_r.T.tocsr(), KS)
        cs = pd.DataFrame({"key": rkeys[col], "s1": s1_q[r]})
        found["c_s1side"] = cs
        del xc_r
        gc.collect()
        mt.stage(f"{c}: c_s1side ({len(cs):,} pairs)")

        rec = pd.DataFrame(index=miss.index)
        for m in METHODS:
            hit = miss.merge(found[m].drop_duplicates(), on=["key", "s1"], how="left", indicator=True)
            rec[m] = (hit["_merge"] == "both").to_numpy()
        rec["union"] = rec[METHODS].any(axis=1)
        rec["slice"], rec["country"] = miss["slice"].to_numpy(), c
        res_rows.append(rec)

        exs = ex[np.isin(ex["key"].to_numpy(), samp)][["key", "s1"]]
        new_all = []
        row = {"country": c}
        for m in ["a_name", "b_addr", "d_sibling", "e_translit"]:
            f = found[m][np.isin(found[m]["key"].to_numpy(), samp)].drop_duplicates()
            f = f.merge(exs, on=["key", "s1"], how="left", indicator=True)
            f = f[f["_merge"] == "left_only"][["key", "s1"]]
            new_all.append(f)
            row[m] = len(f) / N_REC
        fc = cs[np.isin(cs["s1"].to_numpy(), s1_samp)].merge(ex[["key", "s1"]], on=["key", "s1"], how="left", indicator=True)
        row["c_s1side"] = (fc["_merge"] == "left_only").sum() / N_S1 * len(s1_ids) / len(rkeys)
        row["union"] = len(pd.concat(new_all).drop_duplicates()) / N_REC + row["c_s1side"]
        row["current_pruned"] = len(ex) / len(rkeys)
        ex_rows.append(row)
        del ex, top1, sib, found, cs
        gc.collect()
        mt.stage(f"{c}: done")

    res = pd.concat(res_rows, ignore_index=True)
    cols = METHODS + ["union"]
    pd.set_option("display.width", 220)
    print("\nRecovery of missed GT pairs (true S1 not in candidates), fraction recovered:")
    t1 = res.groupby("country")[cols].mean()
    t1.loc["all"] = res[cols].mean()
    t1.insert(0, "pairs", list(res.groupby("country").size()) + [len(res)])
    print(t1.round(4).to_string())
    print("\nper slice (all countries):")
    t2 = res.groupby("slice")[cols].mean()
    t2.insert(0, "pairs", res.groupby("slice").size())
    print(t2.round(4).to_string())
    print("\nper country x slice:")
    t3 = res.groupby(["country", "slice"])[cols].mean()
    t3.insert(0, "pairs", res.groupby(["country", "slice"]).size())
    print(t3.round(4).to_string())
    print("\nExtra candidates per record beyond the pruned v1 lists (union = a,b,d,e deduped + c; c overlap ignored):")
    print(pd.DataFrame(ex_rows).set_index("country").round(3).to_string())
    t3.reset_index().to_csv(WORK / "recovery_diag.tsv", sep="\t", index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
