"""How much text each edition shares with every other edition.

    python -m ltg reuse              # -> results/text/reuse_matrix.csv

For each edition, the distinct sentences of at least MIN_WORDS words (accent-folded,
letters only) of its national student books. Cell (row A, column B) is the share of
A's sentences that also occur, word for word, somewhere in B. Exact matching after
normalisation undercounts reuse (OCR differences, light edits), so read it as a
lower bound: it separates rewritten generations from reprinted ones.
"""
import json
import logging
import re
import unicodedata

import pandas as pd

from .config import PATHS
from .text_analysis import edition_order

log = logging.getLogger("ltg.reuse")
MIN_WORDS = 8
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def _norm(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return " ".join(re.findall(r"[a-zñ]+", s))


def sentences(text, min_words=MIN_WORDS):
    out = set()
    for raw in _SPLIT.split(re.sub(r"(?<!\n)\n(?!\n)", " ", text or "")):
        n = _norm(raw)
        if n.count(" ") + 1 >= min_words:
            out.add(n)
    return out


def edition_sentences(min_words=MIN_WORDS):
    cat = pd.read_csv(PATHS["catalog"], dtype={"edition": str}, keep_default_na=False) if PATHS["catalog"].exists() else None
    state = set()
    if cat is not None and "corpus" in cat:
        state = set(zip(cat.loc[cat["corpus"] == "state", "edition"], cat.loc[cat["corpus"] == "state", "code"]))
    by = {}
    for path in sorted(PATHS["text"].glob("*/*.jsonl")):
        ed, code = path.parent.name, path.stem
        if (ed, code) in state:
            continue
        s = by.setdefault(ed, set())
        with path.open(encoding="utf-8") as f:
            for line in f:
                s |= sentences(json.loads(line).get("text", ""), min_words)
    return by


def run(min_words=MIN_WORDS):
    by = edition_sentences(min_words)
    eds = edition_order(by)
    m = pd.DataFrame(index=eds, columns=eds, dtype=float)
    for a in eds:
        for b in eds:
            m.loc[a, b] = 100 * len(by[a] & by[b]) / len(by[a]) if by[a] else float("nan")
    out = PATHS["results"] / "text"
    out.mkdir(parents=True, exist_ok=True)
    m.round(1).to_csv(out / "reuse_matrix.csv", index_label="edition")
    pd.Series({e: len(by[e]) for e in eds}, name="sentences").to_csv(out / "reuse_sentence_counts.csv", index_label="edition")
    log.info("wrote %s (%d editions)", out / "reuse_matrix.csv", len(eds))
    return m
