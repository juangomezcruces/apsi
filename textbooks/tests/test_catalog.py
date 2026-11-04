"""Catalog parsing. The first fixtures reproduce, in miniature, the markup the live sites
used on 2026-11-03: grade cards on libros.conaliteg.gob.mx, a JavaScript book array on
historico.conaliteg.gob.mx, and links to the encrypted-PDF reader for 2026-27. The
generic cases after them keep the crawler working if the markup changes."""
from ltg import catalog as C

LIVE_CURRENT = """
<li class="grado"><a id="id_6d4ab62f" onclick="">Primer Grado. 2023 - 2024</a></li>
<div class="col-lg-3"><div class="row"><a href="./2023/P1LPM.htm" target="_blank"><div><p class="btn-guinda">
 <img alt="Un libro sin recetas, para la maestra y el maestro. Fase 3." src="./2023/m/P1LPM/000.jpg"/>
 <b>Un libro sin recetas, para la maestra y el maestro. Fase 3.</b><br/> Editorial: SEP<br/> Grado: 1°<br/></p></div></a></div></div>
<div class="col-lg-3"><div class="row"><a href="./2023/P1SDA.htm" target="_blank"><div><p class="btn-guinda">
 <img alt="Nuestros saberes: Libro para alumnos, maestros y familia. " src="./2023/m/P1SDA/000.jpg"/>
 <b>Nuestros saberes: Libro para alumnos, maestros y familia. </b><br/> Editorial: SEP<br/> Grado: 1°<br/></p></div></a></div></div>
<div class="col-lg-3"><div class="row"><a href="./2023/P0CMA.htm" target="_blank"><div><p class="btn-guinda">
 <b>Cartografía de México y el mundo.</b><br/> Editorial: SEP<br/> Grado: 4°<br/></p></div></a></div></div>
<div class="col-lg-3"><div class="row"><a href="./2023/P0CMA.htm" target="_blank"><div><p class="btn-guinda">
 <b>Cartografía de México y el mundo.</b><br/> Editorial: SEP<br/> Grado: 5°<br/></p></div></a></div></div>
<a href="CEAP.pdf">CEAP</a> <a href="https://libros.conaliteg.gob.mx/2019cm.htm">2019</a>
<a href="primaria.html">Primaria</a>
"""

LIVE_2026 = """
<a href="pdf-reader/reader.html?nivel=primaria&ciclo=2026&clave=P5PCA" target="_blank"><p>
 <img alt="Proyectos comunitarios. Quinto grado" src="./2026/m/P5PCA/000.jpg"/>
 <b>Proyectos comunitarios. Quinto grado</b><br/> Editorial: SEP<br/> Grado: 5°<br/></p></a>
"""

LIVE_HISTORICAL_PAGE = """<html><head>
<script src="jquery.js"></script><script src="libros_2023.js"></script><script src="/js/busqueda_01.js"></script>
<script src="https://www.googletagmanager.com/gtag/js?id=G-X"></script></head>
<body><a href="javascript:void(1988)"><img src="./c/1988.jpg"></a></body></html>"""

LIVE_HISTORICAL_JS = """var g = [];
g[1988]=[0,1988,1988,1988,1988,1988,1993];
var libros = [
["MI LIBRO DE 1°, PARTE 1","H1988P1ES108",1988,1],
["CIENCIAS SOCIALES","H1988P5CI124",1988,5],
["MI CUADERNO DE TRABAJO PREESCOLAR","H1988K1ES107",1988,1],
["CIENCIAS SOCIALES","H1972P5CI050",1972,5],
];"""


def test_parse_code():
    assert C.parse_code("P1PAA") == dict(code="P1PAA", hist_year="", level="primaria", grade=1, code_rest="PAA")
    h = C.parse_code("H1988P1ES123")
    assert (h["hist_year"], h["grade"], h["code_rest"]) == ("1988", 1, "ES123")
    assert C.parse_code("primaria") is None and C.parse_code("P1") is None
    assert C.parse_code("Z7822")["level"] == "Z"            # workbook; crawl_edition re-levels it


def test_live_current_cards():
    books = C.extract_books(LIVE_CURRENT, "https://libros.conaliteg.gob.mx/primaria_2023.html")
    assert set(books) == {"P1LPM", "P1SDA", "P0CMA"}         # 2019cm.htm and CEAP.pdf are not books
    assert books["P1SDA"]["viewer_url"] == "https://libros.conaliteg.gob.mx/2023/P1SDA.htm"
    assert books["P1SDA"]["title"] == "Nuestros saberes: Libro para alumnos, maestros y familia"
    assert books["P0CMA"]["grades"] == {4, 5}                # one book listed under several grades
    assert C.audience_from_title(books["P1LPM"]["title"]) == "teacher"
    assert C.audience_from_title(books["P1SDA"]["title"]) == "student"   # 'maestros' but for 'alumnos'


def test_live_2026_reader_links_are_recorded_without_viewer():
    b = C.extract_books(LIVE_2026, "https://libros.conaliteg.gob.mx/primaria.html")["P5PCA"]
    assert (b["viewer_url"], b["access"], b["served_year"], b["grades"]) == ("", "pdf-reader", "2026", {5})


def test_live_historical_script_array():
    base = "https://historico.conaliteg.gob.mx/?g=1988&a=1"
    scripts = C.same_host_scripts(LIVE_HISTORICAL_PAGE, base)
    assert scripts == ["https://historico.conaliteg.gob.mx/libros_2023.js",
                       "https://historico.conaliteg.gob.mx/js/busqueda_01.js"]
    pages = {base: LIVE_HISTORICAL_PAGE, scripts[0]: LIVE_HISTORICAL_JS, scripts[1]: ""}
    rows = C.crawl_edition(dict(edition="1988", seed=base, generation="1988"), None, delay=0, fetch=pages.get)
    by_code = {r["code"]: r for r in rows}
    assert set(by_code) == {"H1988P1ES108", "H1988P5CI124", "H1988K1ES107"}   # 1972 generation filtered out
    r = by_code["H1988P5CI124"]
    assert (r["grade"], r["served_year"], r["period"], r["subject"], r["level"]) == \
        (5, "1988", "gen-1988", "Ciencias sociales", "primaria")
    assert r["viewer_url"] == "https://historico.conaliteg.gob.mx/H1988P5CI124.htm"
    assert by_code["H1988P1ES108"]["subject"] == "Integrado (1°-2°)"
    assert by_code["H1988K1ES107"]["level"] == "preescolar"


def test_state_books_and_odd_levels():
    html = ('<a href="./2022/P3OAX.htm"><b>Oaxaca. La entidad donde vivo</b> Grado: 3°</a>'
            '<a href="./2022/Z7822.htm"><b>Formación Cívica y Ética. Cuaderno de aprendizaje</b> Grado: 5°</a>')
    rows = C.crawl_edition(dict(edition="2022", seed="https://x.mx/primaria_2022.html"), None, delay=0,
                           fetch=lambda u: html if u.endswith(".html") else "")
    by_code = {r["code"]: r for r in rows}
    assert by_code["P3OAX"]["corpus"] == "state" and by_code["Z7822"]["corpus"] == "national"
    assert (by_code["Z7822"]["level"], by_code["Z7822"]["grade"]) == ("primaria", 5)
    assert by_code["Z7822"]["subject"] == "Formación cívica y ética"


# ---- generic markup ---------------------------------------------------------------
GENERIC = """
<div class="card"><a href="2023/P1PAA.htm"><img src="2023/c/P1PAA/000.jpg" alt="Proyectos de Aula. Primer grado"></a></div>
<div class="card"><h3>Nuestros saberes: Libro para alumnos, maestros y familia</h3>
  <a href="2023/P1SDA.htm"><img src="x.jpg"></a> <a href="pdf/P1SDA.pdf">Descargar PDF</a></div>
<a href="#" onclick="window.open('2023/P5PCA.htm')">Proyectos Comunitarios. Quinto grado</a>
<script>var libros = ["2023/P6ENA.htm", "2023/P6ENA.htm"];</script>
<a href="primaria.html">Primaria</a><a href="secundaria_2023.html">Secundaria</a>
"""

GENERIC_NAV = """
<a href="?g=1988&a=1">1o</a><a href="?g=1988&a=2">2o</a><a href="?g=1972&a=1">1972</a>
<a href="H1988P1ES123.htm" title="Mi libro de primero. Recortable"><img src="c/H1988P1ES123/000.jpg"></a>
"""


def test_generic_links():
    books = C.extract_books(GENERIC, "https://libros.conaliteg.gob.mx/primaria_2023.html")
    assert set(books) == {"P1PAA", "P1SDA", "P5PCA", "P6ENA"}
    assert books["P1PAA"]["title"].startswith("Proyectos de Aula")
    assert "Nuestros saberes" in books["P1SDA"]["title"]          # cover with no alt: card text
    assert books["P1SDA"]["pdf_url"] == "https://libros.conaliteg.gob.mx/pdf/P1SDA.pdf"
    assert C.subject_from_title(books["P5PCA"]["title"]) == "Proyectos comunitarios"
    assert C.extract_books('<a href="primaria_2023.html">x</a>', "https://x.mx/") == {}


def test_generic_nav_expand_and_failures():
    base = "https://historico.conaliteg.gob.mx/?g=1988&a=1"
    assert C.nav_links(GENERIC_NAV, base, ["g"]) == [base, "https://historico.conaliteg.gob.mx/?g=1988&a=2"]
    seeds = C.expand_seed(base, {"a": ["1", "2", "3"]})
    assert len(seeds) == 3 and seeds[0] == base

    def fetch(u):
        if u.endswith("a=2"):
            raise IOError("boom")                     # one bad listing page must not lose the rest
        return GENERIC_NAV
    rows = C.crawl_edition(dict(edition="1988", seed=base, expand={"a": ["2", "3"]}, follow_params=["g"]),
                           None, delay=0, fetch=fetch)
    assert len(rows) == 1 and rows[0]["code"] == "H1988P1ES123"


def test_teaching_editions_notes_and_state_books():
    teacher = ["Matemáticas. Primer grado. Docente.", "Español. Libro para el maestro",
               "Fichero didactico (incluye DVD)", "Un libro sin recetas, para la maestra y el maestro. Fase 3"]
    assert all(C.audience_from_title(t) == "teacher" for t in teacher)
    assert C.audience_from_title("Nuestros saberes: Libro para alumnos, maestros y familia") == "student"
    assert C.audience_from_title("Desafíos matemáticos. Libro para el alumno") == "student"
    assert C.audience_from_title("En 1960, los concursos para los libros de sexto grado, fueron declarados "
                                 "desiertos por el jurado") == "note"
    assert C.is_state_book("Aguascalientes. Historia y Geografía", "Historia")
    assert C.is_state_book("México. Historia y Geografía", "Historia")
    assert C.is_state_book("La entidad donde vivo Oaxaca", "La entidad donde vivo")
    assert C.is_state_book("MONOGRAFÍA ESTATAL AGUASCALIENTES", "")
    assert C.is_state_book("Baja California Sur. Historia y Geografía", "Historia")
    assert not C.is_state_book("Atlas de México", "Atlas")
    assert not C.is_state_book("Historia", "Historia")
    assert C.subject_from_title("MI LIBRO DE HISTORIA Y CIVISMO") == "Historia"
    assert C.subject_from_title("MI CUADERNO DE TRABAJO, ARITMÉTICA Y GEOMETRÍA") == "Matemáticas"
    assert C.subject_from_title("EXPLORACIÓN DE LA NATURALEZA Y LA SOCIEDAD") == "Exploración de la naturaleza y la sociedad"
