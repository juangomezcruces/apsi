"""Draw validation samples from the Touche 2025 / ParlaMint training files.

Labels in these files are PARTY attributes copied onto every speech (populism
from expert surveys, orientation from Wikipedia, power from ParlaMint), so the
unit of validation is the SPEAKER, not the speech. The sampler therefore draws
speakers first, a few speeches per speaker, and a few chunks per speech, and
balances across the (populism x orientation x power) cells so that populism
cannot be recovered from ideology or government status alone.

Outputs (all in this folder):
  touche_criterion.csv     -- AT ES FR IT NL, criterion validity vs expert populism
  touche_gb_discrim.csv    -- GB, discriminant test within populism level 3
  touche_sample_meta.json  -- the parameters and resulting cell counts

Each output row is one CHUNK, ready for run(): id, text_en, plus metadata.
"""
import csv
import hashlib
import json
import re

import numpy as np
import pandas as pd

SEED = 2025
COUNTRIES = ["at", "es", "fr", "it", "nl"]

MIN_WORDS = 40          # below this a speech is a procedural interjection
SPEAKERS_PER_CELL = 12  # cap per (populism, orientation, power) cell
SPEECHES_PER_SPEAKER = 4
CHUNKS_PER_SPEECH = 3
CHUNK_WORDS = 150       # ~200 tokens; the NLI model truncates at 512
DEV_SHARE = 0.3         # share of SPEAKERS held out for any tuning

GB_SPEAKERS_PER_CELL = 60


def load(cc):
    d = pd.read_csv(f"{cc}-train.tsv", sep="\t", quoting=csv.QUOTE_NONE,
                    on_bad_lines="skip", low_memory=False)
    d["country"] = cc
    d["text_en"] = d["text_en"].fillna("").astype(str)
    d["words"] = d["text_en"].str.split().str.len()
    return d


def clean(d):
    d = d[d.populism.notna()]
    d = d[d.words >= MIN_WORDS]
    # party switchers carry two populism values; their label is ambiguous
    switch = d.groupby("speaker").populism.nunique()
    return d[~d.speaker.isin(switch[switch > 1].index)].copy()


SENT = re.compile(r"(?<=[.!?])\s+")


def chunks(text, rng):
    """Sentence-aligned ~CHUNK_WORDS chunks; keep CHUNKS_PER_SPEECH at random.
    Random rather than first-k: openings are salutations ('Mr President, dear
    colleagues...') and carry no position."""
    out, cur, n = [], [], 0
    for s in SENT.split(text):
        w = len(s.split())
        if n and n + w > CHUNK_WORDS:
            out.append(" ".join(cur)); cur, n = [], 0
        cur.append(s); n += w
    if cur and (n >= MIN_WORDS or not out):
        out.append(" ".join(cur))
    if len(out) > CHUNKS_PER_SPEECH:
        keep = sorted(rng.choice(len(out), CHUNKS_PER_SPEECH, replace=False))
        out = [out[i] for i in keep]
    return out


def split_of(speaker):
    h = int(hashlib.md5(str(speaker).encode()).hexdigest(), 16) % 1000
    return "dev" if h < DEV_SHARE * 1000 else "test"


def draw(d, cells, spk_per_cell, rng):
    rows = []
    for key, g in d.groupby(cells, dropna=False):
        spk = g.speaker.unique()
        spk = rng.choice(spk, min(len(spk), spk_per_cell), replace=False)
        for s in spk:
            sp = g[g.speaker == s]
            sp = sp.sample(min(len(sp), SPEECHES_PER_SPEAKER),
                           random_state=int(rng.integers(1 << 31)))
            for _, r in sp.iterrows():
                for k, c in enumerate(chunks(r.text_en, rng)):
                    rows.append({
                        "id": f"{r.id}_c{k}", "speech_id": r.id,
                        "speaker": r.speaker, "country": r.country,
                        "populism": r.populism, "orientation": r.orientation,
                        "power": r.power, "split": split_of(r.speaker),
                        "faction": f"{r.country}-pop{int(r.populism)}",
                        "text_en": c,
                    })
    return pd.DataFrame(rows)


def report(df, by):
    t = (df.groupby(by, dropna=False)
           .agg(chunks=("id", "size"), speeches=("speech_id", "nunique"),
                speakers=("speaker", "nunique"))
           .reset_index())
    return t


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)

    crit = pd.concat([clean(load(cc)) for cc in COUNTRIES], ignore_index=True)
    crit_s = draw(crit, ["country", "populism", "orientation", "power"],
                  SPEAKERS_PER_CELL, rng)
    crit_s.to_csv("touche_criterion.csv", index=False)

    gb = clean(load("gb"))
    gb = gb[gb.populism == 3]           # hold populism fixed
    gb_s = draw(gb, ["orientation", "power"], GB_SPEAKERS_PER_CELL, rng)
    gb_s["role"] = np.select(
        [(gb_s.orientation == 0) & (gb_s.power == 0),
         (gb_s.orientation == 1) & (gb_s.power == 1),
         (gb_s.orientation == 1) & (gb_s.power == 0)],
        ["left-opposition", "right-government", "right-opposition"], "other")
    gb_s = gb_s[gb_s.role != "other"]
    gb_s.to_csv("touche_gb_discrim.csv", index=False)

    meta = {k: v for k, v in globals().items() if k.isupper()}
    meta["criterion_cells"] = report(
        crit_s, ["country", "populism", "orientation", "power"]).to_dict("records")
    meta["gb_cells"] = report(gb_s, ["role"]).to_dict("records")
    json.dump(meta, open("touche_sample_meta.json", "w"), indent=1, default=str)

    print("CRITERION SAMPLE")
    print(report(crit_s, ["country", "populism"]).to_string(index=False))
    print(f"\ntotal: {len(crit_s)} chunks, {crit_s.speech_id.nunique()} speeches, "
          f"{crit_s.speaker.nunique()} speakers | "
          f"dev speakers {crit_s[crit_s.split=='dev'].speaker.nunique()}, "
          f"test {crit_s[crit_s.split=='test'].speaker.nunique()}")
    print("\nGB DISCRIMINANT SAMPLE (populism fixed at 3)")
    print(report(gb_s, ["role"]).to_string(index=False))
    print(f"\ntotal: {len(gb_s)} chunks")
