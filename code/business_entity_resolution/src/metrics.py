"""Per-entity macro F0.5, the competition metric.

F0.5 is computed per S1 entity and averaged over all S1 entities. Empty truth with an empty
prediction scores 1.0; empty truth with any prediction, or non-empty truth with an empty
prediction, scores 0.0. Also reports micro precision / recall for threshold tuning.
"""

from __future__ import annotations

import numpy as np

BETA2 = 0.25


def parse_ids(s: str) -> set[str]:
    return {x for x in (t.strip() for t in s.split(",")) if x} if s else set()


def f05(true: set[str], pred: set[str]) -> float:
    if not true:
        return 1.0 if not pred else 0.0
    tp = len(true & pred)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def macro_f05(truth: dict[str, set[str]], pred: dict[str, set[str]]) -> dict[str, float]:
    n = len(truth)
    tot = tp = npred = ntrue = 0.0
    for k, t in truth.items():
        p = pred.get(k, set())
        tot += f05(t, p)
        tp += len(t & p)
        npred += len(p)
        ntrue += len(t)
    return {
        "macro_f05": tot / n if n else 0.0,
        "micro_precision": tp / npred if npred else 1.0,
        "micro_recall": tp / ntrue if ntrue else 1.0,
        "n": n,
    }


def macro_f05_counts(ntrue, npred, tp) -> dict[str, float]:
    ntrue, npred, tp = (np.asarray(a, dtype=np.float64) for a in (ntrue, npred, tp))
    with np.errstate(divide="ignore", invalid="ignore"):
        p, r = tp / npred, tp / ntrue
        f = np.where(tp > 0, (1 + BETA2) * p * r / (BETA2 * p + r), 0.0)
    f = np.where(ntrue == 0, (npred == 0).astype(np.float64), f)
    return {"macro_f05": float(f.mean()) if len(f) else 0.0,
            "micro_precision": float(tp.sum() / npred.sum()) if npred.sum() else 1.0,
            "micro_recall": float(tp.sum() / ntrue.sum()) if ntrue.sum() else 1.0,
            "n": len(f)}


def _self_test() -> None:
    t = {"S2-00047", "S3-00812"}
    p = {"S2-00047", "S2-00193", "S3-00812"}
    assert round(f05(t, p), 3) == 0.714, f05(t, p)
    assert f05(set(), set()) == 1.0
    assert f05(set(), {"S2-1"}) == 0.0
    assert f05({"S2-1"}, set()) == 0.0
    assert parse_ids("") == set() and parse_ids("S2-1,S3-2") == {"S2-1", "S3-2"}
    m = macro_f05({"a": t, "b": set()}, {"a": p})
    assert abs(m["macro_f05"] - (f05(t, p) + 1.0) / 2) < 1e-12
    v = macro_f05_counts([2, 0, 1, 0], [3, 0, 0, 1], [2, 0, 0, 0])
    assert abs(v["macro_f05"] - (f05(t, p) + 1.0) / 4) < 1e-12, v
    print(f"metrics self-test OK: README example F0.5 = {f05(t, p):.3f}")


if __name__ == "__main__":
    _self_test()
