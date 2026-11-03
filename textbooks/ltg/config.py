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

EDITIONS = [
    # The historical archive (`g` = generation, `a` = grade) builds its list in
    # JavaScript from ["TITLE","CODE",generation,grade] arrays in a same-host
    # script; the crawler reads those and keeps the given generation.
    dict(edition="1988", seed="https://historico.conaliteg.gob.mx/?g=1988&a=1", generation="1988"),
    dict(edition="2022", seed="https://libros.conaliteg.gob.mx/primaria_2022.html"),
    dict(edition="2023", seed="https://libros.conaliteg.gob.mx/primaria_2023.html"),
    dict(edition="2024", seed="https://libros.conaliteg.gob.mx/primaria_2024.html"),
    dict(edition="2025", seed="https://libros.conaliteg.gob.mx/primaria_2025.html"),
    dict(edition="current", seed="https://libros.conaliteg.gob.mx/primaria.html"),
]

# Analytic periods. The 2022 books predate the Nueva Escuela Mexicana (NEM)
# textbooks first distributed for 2023-24; 1988 is the archive's '1988 generation'.
PERIOD = {"1988": "gen-1988", "2022": "pre-NEM", "2023": "NEM", "2024": "NEM",
          "2025": "NEM", "current": "NEM"}

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
