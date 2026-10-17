"""Hand-written Indic -> Latin transliteration and a script-neutral consonant skeleton.

The Devanagari table is reused for the other Brahmic blocks (Bengali to Malayalam), which share
the ISCII-derived layout. skeleton() drops vowels and merges aspirates and voicing, so Latin
and Indic spellings of a name map to similar strings. Used as a matcher feature and as the
fallback for tokens missing from the learned Indic dictionary.
"""

from __future__ import annotations

import re

_DEV = {
    0x01: "n", 0x02: "n", 0x03: "h",
    0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri", 0x0C: "li",
    0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au",
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "n",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v",
    0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x3C: "", 0x3D: "",
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri",
    0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au",
    0x4D: "",
    0x58: "k", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
    0x60: "ri", 0x61: "li", 0x62: "li", 0x63: "li",
    0x50: "om",
    **{0x66 + d: str(d) for d in range(10)},
}
_TABLE: dict[int, str] = {}
for _base in range(0x0900, 0x0D80, 0x80):
    for _off, _lat in _DEV.items():
        _TABLE[_base + _off] = _lat

_TABLE.update({0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l", 0x0D7E: "l"})
_TABLE[0x0B9A] = "s"
_MAL_TT = "റ്റ"

_INDIC_RE = re.compile(r"[ऀ-ൿ]")
_DIGRAPHS = [("chh", "c"), ("ch", "c"), ("sh", "s"), ("ph", "f"), ("bh", "b"), ("kh", "k"),
             ("gh", "g"), ("th", "t"), ("dh", "d"), ("jh", "j"), ("ck", "k"),
             ("hp", "f")]
_SINGLE = str.maketrans({"q": "k", "z": "j", "w": "v", "x": "ks", "d": "t", "b": "p", "g": "k"})
_VOWELS_RE = re.compile(r"(?<=\w)[aeiouy]+")
_REPEAT_RE = re.compile(r"(.)\1+")


def has_indic(s: str) -> bool:
    return _INDIC_RE.search(s) is not None


def translit(s: str) -> str:
    if not has_indic(s):
        return s
    return s.replace(_MAL_TT, "tt").translate(_TABLE)


def _skel(s: str) -> str:
    s = translit(s)
    for a, b in _DIGRAPHS:
        s = s.replace(a, b.upper())
    s = s.replace("c", "k").lower().translate(_SINGLE)
    s = _VOWELS_RE.sub("", s)
    return _REPEAT_RE.sub(r"\1", s)


def _legal_skeletons() -> set[str]:
    from normalize import LEGAL
    words = set(LEGAL) | {"praivet", "limitet", "limited", "privet", "pra", "li"}
    return {_skel(w) for w in words}


LEGAL_SK = _legal_skeletons()


def skeleton(s: str) -> str:
    toks = _skel(s).split()
    core = [t for t in toks if t not in LEGAL_SK]
    return " ".join(core or toks)
