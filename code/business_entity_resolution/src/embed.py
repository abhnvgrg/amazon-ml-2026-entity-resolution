"""EXPERIMENT (sub-07, abandoned): encode record texts with multilingual-e5-small.

Runs in the optional Python 3.12 GPU environment (requirements-dense.txt). The model is
loaded offline from a local directory (--model, or env ER_E5, default: the local Hugging Face
cache). Reads the text shards written by export_texts.py and writes L2-normalized float16
embeddings to work/embeddings/; resumable via .done markers.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[3]
EMB = Path(os.environ.get("ER_WORK", ROOT / "work")) / "embeddings"
MODEL = Path(os.environ.get("ER_E5", Path.home() / ".cache" / "huggingface" / "hub" / "models--intfloat--multilingual-e5-small"))
PREFIX = "query: "
DIM = 384


def model_dir(base: Path = MODEL) -> Path:
    snaps = sorted((base / "snapshots").iterdir()) if (base / "snapshots").exists() else []
    return snaps[-1] if snaps else base


def load_model(max_len: int, base: Path = MODEL) -> SentenceTransformer:
    m = SentenceTransformer(str(model_dir(base)), device="cuda", local_files_only=True)
    m.half()
    m.max_seq_length = max_len
    return m


def shards(splits: list[str]) -> list[tuple[str, str, str]]:
    out = []
    for split in splits:
        for p in sorted(EMB.glob(f"{split}_*_S1.txt")):
            c = p.name[len(split) + 1:-len("_S1.txt")]
            out += [(split, c, "S1"), (split, c, "R")]
    return out


def encode_shard(m: SentenceTransformer, split: str, c: str, kind: str, batch: int, limit: int, log_every: int) -> float:
    texts = (EMB / f"{split}_{c}_{kind}.txt").read_text(encoding="utf-8").split("\n")[:-1]
    keys = np.load(EMB / f"{split}_{c}_{kind}_keys.npy")
    assert len(texts) == len(keys)
    n = min(limit, len(texts)) if limit else len(texts)
    tag = f"_limit{limit}" if limit else ""
    out_path = EMB / f"{split}_{c}_{kind}{tag}.npy"
    done = EMB / f"{split}_{c}_{kind}{tag}.done"
    if done.exists():
        print(f"{split} {c} {kind}: done already, skipping", flush=True)
        return 0.0
    emb = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(n, DIM))
    t0 = last = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    step = log_every
    for lo in range(0, n, step):
        hi = min(lo + step, n)
        v = m.encode([PREFIX + t for t in texts[lo:hi]], batch_size=batch, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)
        emb[lo:hi] = v.astype(np.float16)
        now = time.perf_counter()
        print(f"{split} {c} {kind}: {hi:,}/{n:,} records, {(hi - lo) / (now - last):,.0f} rec/s "
              f"(avg {hi / (now - t0):,.0f}), peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
        last = now
    emb.flush()
    del emb
    if tag:
        np.save(EMB / f"{split}_{c}_{kind}{tag}_keys.npy", keys[:n])
    done.write_text(f"{n}\n", encoding="utf-8")
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["train", "test"])
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="timed test: encode only the first N texts of the first shard")
    ap.add_argument("--log-every", type=int, default=10_000)
    ap.add_argument("--model", type=Path, default=MODEL, help="local e5 model directory (HF cache repo dir or snapshot)")
    a = ap.parse_args()
    m = load_model(a.max_len, a.model)
    print(f"model {model_dir(a.model)} on {torch.cuda.get_device_name(0)}, fp16, max_len {a.max_len}, batch {a.batch}", flush=True)
    todo = shards(a.splits)
    if a.limit:
        split, c, _ = todo[0]
        todo = [(split, c, "R")]
    for split, c, kind in todo:
        sec = encode_shard(m, split, c, kind, a.batch, a.limit, a.log_every)
        print(f"{split} {c} {kind}: finished in {sec / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
