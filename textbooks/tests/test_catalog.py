"""Catalog parsing against synthetic markup in the shapes such catalogs commonly take
(plain links, cover images with alt text, onclick handlers, script arrays). The live
pages have not been seen from the development environment: re-check catalog.csv."""
from ltg import catalog as C

CURRENT = """
<html><body>
<div class="card"><a href="2023/P1PAA.htm"><img src="2023/c/P1PAA/000.jpg" alt="Proyectos de Aula. Primer grado"></a></div>
<div class="card"><h3>Nuestros saberes: Libro para alumnos, maestros y familia</h3>
  <a href="2023/P1SDA.htm"><img src="x.jpg"></a> <a href="pdf/P1SDA.pdf">Descargar PDF</a></div>
<a href="#" onclick="window.open('2023/P5PCA.htm')">Proyectos Comunitarios. Quinto grado</a>
<script>var libros = ["2023/P6ENA.htm", "2023/P6ENA.htm"];</script>
<a href="primaria.html">Primaria</a><a href="secundaria_2023.html">Secundaria</a>
</body></html>"""

HISTORICAL = """
<a href="?g=1988&a=1">1o</a><a href="?g=1988&a=2">2o</a><a href="?g=1972&a=1">1972</a>
<a href="H1988P1ES123.htm" title="Mi libro de primero. Recortable"><img src="c/H1988P1ES123/000.jpg"></a>
"""


def test_parse_code():
    assert C.parse_code("P1PAA") == dict(code="P1PAA", hist_year="", level="primaria", grade=1, code_rest="PAA")
    h = C.parse_code("H1988P1ES123")
    assert (h["hist_year"], h["grade"], h["code_rest"]) == ("1988", 1, "ES123")
    assert C.parse_code("primaria") is None and C.parse_code("P1") is None


def test_extract_books_current():
    books = C.extract_books(CURRENT, "https://libros.conaliteg.gob.mx/primaria_2023.html")
    assert set(books) == {"P1PAA", "P1SDA", "P5PCA", "P6ENA"}
    assert books["P1PAA"]["viewer_url"] == "https://libros.conaliteg.gob.mx/2023/P1PAA.htm"
    assert books["P1PAA"]["title"].startswith("Proyectos de Aula")
    assert "Nuestros saberes" in books["P1SDA"]["title"]          # cover with no alt: card text
    assert books["P1SDA"]["pdf_url"] == "https://libros.conaliteg.gob.mx/pdf/P1SDA.pdf"
    assert C.subject_from_title(books["P5PCA"]["title"]) == "Proyectos comunitarios"


def test_html_pages_are_not_books():
    assert C.extract_books('<a href="primaria_2023.html">x</a>', "https://x.mx/") == {}


def test_historical_nav_and_crawl():
    base = "https://historico.conaliteg.gob.mx/?g=1988&a=1"
    nav = C.nav_links(HISTORICAL, base, ["g"])
    assert nav == ["https://historico.conaliteg.gob.mx/?g=1988&a=1", "https://historico.conaliteg.gob.mx/?g=1988&a=2"]
    seeds = C.expand_seed(base, {"a": ["1", "2", "3"]})
    assert len(seeds) == 3 and seeds[0] == base

    pages = {u: HISTORICAL for u in seeds}
    spec = dict(edition="1988", seed=base, expand={"a": ["1", "2", "3"]}, follow_params=["g"])
    rows = C.crawl_edition(spec, session=None, delay=0, fetch=lambda u: pages.get(u, ""))
    assert len(rows) == 1                                        # duplicates across listing pages dropped
    r = rows[0]
    assert (r["code"], r["grade"], r["served_year"], r["period"]) == ("H1988P1ES123", 1, "1988", "pre-1993")
    assert r["viewer_url"] == "https://historico.conaliteg.gob.mx/H1988P1ES123.htm"
    assert r["subject"] == "Lengua"


def test_failed_listing_page_is_skipped():
    def fetch(u):
        if u.endswith("a=2"):
            raise IOError("boom")
        return HISTORICAL
    spec = dict(edition="1988", seed="https://historico.conaliteg.gob.mx/?g=1988&a=1",
                expand={"a": ["2"]}, follow_params=[])
    assert len(C.crawl_edition(spec, None, delay=0, fetch=fetch)) == 1
