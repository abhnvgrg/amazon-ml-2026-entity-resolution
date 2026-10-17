"""Hand-written text normalization shared by train and test (no external data).

norm_name(s)    -> lowercased, accent-folded business name with abbreviations expanded
name_core(n)    -> normalized name with legal-suffix tokens removed (falls back to n)
norm_address(s) -> normalized address (street-type abbreviations expanded, punctuation removed)
Indic-script characters, including vowel signs and viramas, are kept as-is.
"""

from __future__ import annotations

import re
import unicodedata

_KEEP = "̀-ͯऀ-෿"
_PUNCT_RE = re.compile(rf"[^\w\s{_KEEP}]|_")
_DOTTED_RE = re.compile(r"(?<=\b\w)\.(?=\w\b)")
_WS_RE = re.compile(r"\s+")

NAME_MAP = {
    "pvt": "private", "pvt.": "private", "ltd": "limited", "incorporated": "inc",
    "corp": "corporation", "co": "company", "cie": "compagnie", "ets": "etablissements",
    "l l c": "llc",
}
ADDR_MAP = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue", "dr": "drive",
    "ln": "lane", "blvd": "boulevard", "bd": "boulevard", "r": "rue", "imp": "impasse",
    "pl": "place",
}
ADDR_DROP = {"no", "nº", "n°"}

LEGAL = {
    "private", "limited", "llc", "llp", "inc", "corporation", "company", "compagnie",
    "plc", "pllc", "lp", "pc", "ltda", "gmbh", "sarl", "sas", "sasu", "eurl", "sa", "sci",
    "snc", "prv", "pvt", "ltd", "public", "प्राइवेट", "लिमिटेड", "प्रा", "लि",
}


class _AccentTable(dict):

    def __missing__(self, cp: int) -> str:
        ch = chr(cp)
        if 0x0900 <= cp <= 0x0DFF:
            v = ch
        elif 0x0300 <= cp <= 0x036F:
            v = ""
        else:
            v = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
        self[cp] = v
        return v


_ACCENTS = _AccentTable()


def _strip_latin_accents(s: str) -> str:
    return s.translate(_ACCENTS)


def _base(s: str) -> str:
    s = s.lower()
    if not s.isascii():
        s = _strip_latin_accents(s)
    s = s.replace("&", " and ")
    s = _DOTTED_RE.sub("", s)
    s = _PUNCT_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def norm_name(s: str) -> str:
    return " ".join(NAME_MAP.get(t, t) for t in _base(s).split())


def name_core(n: str) -> str:
    core = " ".join(t for t in n.split() if t not in LEGAL)
    return core or n


def norm_address(s: str) -> str:
    toks = _base(s).split()
    out = []
    for i, t in enumerate(toks):
        if t in ADDR_DROP:
            continue
        if t == "n" and i + 1 < len(toks) and toks[i + 1][:1].isdigit():
            continue
        out.append(ADDR_MAP.get(t, t))
    return " ".join(out)


def normalize_df(df, name_col="business_name", addr_col="business_address"):
    names = [norm_name(x) for x in df[name_col].tolist()]
    df = df.assign(name=names, core=[name_core(x) for x in names],
                   addr=[norm_address(x) for x in df[addr_col].tolist()])
    return df
