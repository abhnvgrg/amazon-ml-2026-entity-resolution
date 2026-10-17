"""Shared paths, TSV IO, the grouped validation split and memory tracking.

Paths resolve relative to the repo root and can be overridden with environment variables:
ER_DATA (default dataset/), ER_WORK (default work/, all caches) and ER_OUT (default output/).
`is_val` is the deterministic hash split that puts ~20% of S1 entities, with all of their
records, in validation.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import gc
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("ER_DATA", ROOT / "dataset"))
WORK = Path(os.environ.get("ER_WORK", ROOT / "work"))
OUT = Path(os.environ.get("ER_OUT", ROOT / "output"))
RD = dict(sep="\t", dtype=str, keep_default_na=False)
SEED = 42
SRCS = ("S1", "S2", "S3")


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False, encoding="utf-8", lineterminator="\n")


def is_val(s1, frac: int = 5):
    h = (np.asarray(s1, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)
    return (h % np.uint64(frac)) == 0


def id_str(src: str, i) -> str:
    return f"{src}-{i}"


class MemTracker:

    def __init__(self, log=print):
        self.p = psutil.Process()
        self.peak = 0
        self.log = log
        self.t0 = time.perf_counter()
        threading.Thread(target=self._run, daemon=True).start()

    def _now(self) -> int:
        m = self.p.memory_info()
        return getattr(m, "private", m.rss)

    def _run(self):
        while True:
            self.peak = max(self.peak, self._now())
            time.sleep(0.1)

    def stage(self, name: str) -> dict:
        gc.collect()
        dt = time.perf_counter() - self.t0
        rec = {"stage": name, "sec": round(dt, 1), "peak_gb": round(max(self.peak, self._now()) / 2**30, 2)}
        self.log(f"[mem] {name}: {rec['sec']}s, peak {rec['peak_gb']} GB")
        self.peak, self.t0 = self._now(), time.perf_counter()
        return rec
