"""EXPERIMENT (sub-07, abandoned): exact dense top-10 retrieval with e5 embeddings.

Runs in the optional Python 3.12 GPU environment (requirements-dense.txt). For each record,
an fp16 GPU matrix product against the same country's S1 embeddings (S1 order hash-permuted
for tie-breaking). Output: work/dense/{split}_{country}_pairs.npz. On the val sample the dense
candidates added only +0.0014 blocking recall, so they were not adopted.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
WORK = Path(os.environ.get("ER_WORK", ROOT / "work"))
EMB = WORK / "embeddings"
OUT = WORK / "dense"


def mix64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.int64).astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def shards(splits: list[str]) -> list[tuple[str, str]]:
    out = []
    for split in splits:
        for p in sorted(EMB.glob(f"{split}_*_S1.done")):
            c = p.name[len(split) + 1:-len("_S1.done")]
            if (EMB / f"{split}_{c}_R.done").exists():
                out.append((split, c))
    return out


@torch.inference_mode()
def search(split: str, c: str, k: int, q_chunk: int, s_block: int) -> None:
    s1_ids = np.load(EMB / f"{split}_{c}_S1_keys.npy")
    rkeys = np.load(EMB / f"{split}_{c}_R_keys.npy")
    perm = np.argsort(mix64(s1_ids), kind="stable")
    sp = s1_ids[perm]
    S = torch.from_numpy(np.load(EMB / f"{split}_{c}_S1.npy")[perm]).cuda()
    R = np.load(EMB / f"{split}_{c}_R.npy", mmap_mode="r")
    assert R.shape[0] == len(rkeys) and S.shape[0] == len(s1_ids)
    del perm
    kk = min(k, S.shape[0])
    vals = np.empty((len(rkeys), kk), np.float16)
    idx = np.empty((len(rkeys), kk), np.int64)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for lo in range(0, len(rkeys), q_chunk):
        q = torch.from_numpy(np.ascontiguousarray(R[lo:lo + q_chunk])).cuda()
        bv, bi = [], []
        for so in range(0, S.shape[0], s_block):
            sc = q @ S[so:so + s_block].T
            v, i = torch.topk(sc, min(kk, sc.shape[1]), dim=1)
            bv.append(v)
            bi.append(i + so)
        v, j = torch.topk(torch.cat(bv, 1), kk, dim=1)
        i = torch.gather(torch.cat(bi, 1), 1, j)
        vals[lo:lo + len(q)] = v.cpu().numpy()
        idx[lo:lo + len(q)] = sp[i.cpu().numpy()]
        if (lo // q_chunk) % 100 == 0:
            done = lo + len(q)
            print(f"{split} {c}: {done:,}/{len(rkeys):,} records, {done / (time.perf_counter() - t0):,.0f} rec/s, "
                  f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    del S
    torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{split}_{c}_pairs.npz", record_id=rkeys, s1_id=idx, cos_d=vals)
    print(f"{split} {c}: {len(rkeys) * kk:,} pairs in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


def resave_compact(split: str, c: str, k: int = 10) -> None:
    p = OUT / f"{split}_{c}_pairs.npz"
    z = np.load(p)
    if z["s1_id"].ndim == 2:
        print(f"{split} {c}: already compact", flush=True)
        return
    n = len(z["record_id"]) // k
    rank = z["rank_d"].reshape(n, k)
    assert (rank == np.arange(k)).all()
    rid = z["record_id"][::k].copy()
    s1 = z["s1_id"].reshape(n, k)
    cos = z["cos_d"].astype(np.float16).reshape(n, k)
    del rank
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez(tmp, record_id=rid, s1_id=s1, cos_d=cos)
    del z
    tmp.replace(p)
    print(f"{split} {c}: resaved {n:,} records x {k} compact", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["train", "test"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--q-chunk", type=int, default=4096)
    ap.add_argument("--s-block", type=int, default=131072)
    ap.add_argument("--resave", nargs=2, metavar=("SPLIT", "COUNTRY"), help="convert an old flat pairs.npz to compact")
    a = ap.parse_args()
    if a.resave:
        resave_compact(*a.resave, a.k)
        return
    for split, c in shards(a.splits):
        if (OUT / f"{split}_{c}_pairs.npz").exists():
            print(f"{split} {c}: pairs exist, skipping", flush=True)
            continue
        search(split, c, a.k, a.q_chunk, a.s_block)


if __name__ == "__main__":
    main()
