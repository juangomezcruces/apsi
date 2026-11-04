"""Corpus definition and paths.

Six catalog pages, one per edition. Each edition is labelled by the school
year its catalog page names; `current` is whatever `primaria.html` serves on
the day the crawl runs (the crawler records the year folder each book actually
lives in, so the label can be fixed after the fact).

URL scheme (documented by github.com/jlochoap/LTGbasic, not yet verified
against the live site from this environment, see README):

    historical viewer   https://historico.conaliteg.gob.mx/H1960P1ES002.htm
    historical pages    https://historico.conaliteg.gob.mx/c/H1960P1ES002/034.jpg
    current viewer      https://libros.conaliteg.gob.mx/2025/S3LEA.htm
    current pages       https://libros.conaliteg.gob.mx/2025/c/S3LEA/024.jpg

i.e. page images live in `c/<CODE>/<nnn>.jpg` next to the viewer page.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("LTG_DATA", ROOT / "data"))

# Every generation in the historical archive (`g` = generation, `a` = grade). The
# archive builds its list in JavaScript from ["TITLE","CODE",generation,grade]
# arrays in a same-host script; the crawler reads those and keeps one generation.
HIST_GENERATIONS = ["1960", "1962", "1972", "1982", "1988", "1993", "2008", "2011", "2014", "2018", "2019"]

EDITIONS = [dict(edition=g, seed=f"https://historico.conaliteg.gob.mx/?g={g}&a=1", generation=g)
            for g in HIST_GENERATIONS] + [
    dict(edition="2022", seed="https://libros.conaliteg.gob.mx/primaria_2022.html"),
    dict(edition="2023", seed="https://libros.conaliteg.gob.mx/primaria_2023.html"),
    dict(edition="2024", seed="https://libros.conaliteg.gob.mx/primaria_2024.html"),
    dict(edition="2025", seed="https://libros.conaliteg.gob.mx/primaria_2025.html"),
    dict(edition="current", seed="https://libros.conaliteg.gob.mx/primaria.html"),
]

# Editions in chronological order (this order is used in every table and figure).
# Historical generations keep their own label; 2022 predates the Nueva Escuela
# Mexicana (NEM) books first distributed for 2023-24.
PERIOD = {**{g: f"gen-{g}" for g in HIST_GENERATIONS},
          "2022": "pre-NEM", "2023": "NEM", "2024": "NEM", "2025": "NEM", "current": "NEM"}

# Coarser groups, used only to stratify the hand-coding samples (decades of the archive;
# 2023-2025 are reprints of one set of books). Descriptive labels, not periodisation claims.
ERA = {"1960": "1960s", "1962": "1960s", "1972": "1970s-80s", "1982": "1970s-80s",
       "1988": "1988-93", "1993": "1988-93", "2008": "2008-14", "2011": "2008-14", "2014": "2008-14",
       "2018": "2018-22", "2019": "2018-22", "2022": "2018-22",
       "2023": "NEM", "2024": "NEM", "2025": "NEM", "current": "NEM"}

HEADERS = {
    "User-Agent": "ltg-research/0.1 (academic text analysis of public textbooks)",
    "Accept-Language": "es-MX,es;q=0.9",
}

PATHS = dict(
    catalog=DATA / "catalog.csv",
    pages=DATA / "pages",          # pages/<edition>/<CODE>/<nnn>.jpg
    pdf=DATA / "pdf",              # pdf/<edition>/<CODE>.pdf
    text=DATA / "text",            # text/<edition>/<CODE>.jsonl  (one record per page)
    boxes=DATA / "boxes",          # boxes/<edition>/<CODE>/<nnn>.tsv.gz (word boxes)
    results=DATA / "results",
    visual=DATA / "visual",
)


def set_data_root(root):
    """Point every stage at another data directory (tests, parallel corpora)."""
    global DATA
    DATA = Path(root)
    os.environ["LTG_DATA"] = str(DATA)        # worker processes started by spawn re-read it
    for key, sub in dict(catalog="catalog.csv", pages="pages", pdf="pdf", text="text",
                         boxes="boxes", results="results", visual="visual").items():
        PATHS[key] = DATA / sub
