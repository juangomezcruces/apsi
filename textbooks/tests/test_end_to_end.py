"""Every CLI stage in order, against a local mock of both servers with synthetic books."""
import functools
import shutil
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd
import pytest

from ltg import catalog
from ltg.__main__ import main as cli
from ltg.config import PATHS

from conftest import make_page

pytestmark = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")

BOOKS = {
    "hist/H1988P5CS001": ["El pueblo de México luchó contra la dictadura.",
                          "Los campesinos defendieron la tierra.",
                          "Vivo en un pueblo pequeño cerca de la ciudad."],
    "cur/2023/P5PCA": ["Los pueblos originarios conservan sus lenguas.",
                       "La comunidad decide en asamblea.",
                       "Respetamos la diversidad de opiniones."],
}


@pytest.fixture
def site(tmp_path):
    root = tmp_path / "site"
    (root / "hist").mkdir(parents=True)
    (root / "hist/index.html").write_text(
        '<a href="?g=1988&a=2">2o</a><a href="H1988P5CS001.htm" title="Ciencias Sociales. Quinto grado">x</a>')
    (root / "cur/2023").mkdir(parents=True)
    (root / "cur/primaria_2023.html").write_text(
        '<a href="2023/P5PCA.htm"><img alt="Proyectos Comunitarios. Quinto grado"></a>')
    for path, lines in BOOKS.items():
        (root / f"{path}.htm").write_text("<script>var totalPaginas = 2;</script>")
        d = root / path.rsplit("/", 1)[0] / "c" / path.rsplit("/", 1)[1]
        d.mkdir(parents=True)
        for n in (1, 2):
            make_page(lines, seed=n, illustration=(n == 1)).save(d / f"{n:03d}.jpg", quality=90)

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(root)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_pipeline(site, data_root, monkeypatch):
    monkeypatch.setattr(catalog, "EDITIONS", [
        dict(edition="1988", seed=f"{site}/hist/?g=1988&a=1", expand={"a": ["1", "2"]}, follow_params=["g"]),
        dict(edition="2023", seed=f"{site}/cur/primaria_2023.html")])
    cat = cli(["catalog", "--delay", "0"])
    assert sorted(cat["code"]) == ["H1988P5CS001", "P5PCA"]
    assert set(cat["subject"]) == {"Ciencias sociales", "Proyectos comunitarios"}

    res = cli(["download", "--delay", "0"])
    assert all(m["complete"] and m["n_pages"] == 2 for m in res)

    cli(["extract", "--workers", "2"])
    cli(["text"])
    cli(["report"])
    cli(["sample"])
    report = (PATHS["results"] / "text" / "report.md").read_text()
    assert "Ciencias sociales" in report and "Proyectos comunitarios" in report   # by-subject table
    ptok = pd.read_csv(PATHS["results"] / "text" / "people_tokens.csv.gz", dtype={"edition": str})
    # the page is OCR'd twice per book (two pages share the text), so each sentence appears twice
    senses = ptok[ptok["lemma"] == "pueblo"].groupby(["edition", "sense"]).size().to_dict()
    assert senses == {("1988", "people_political"): 2, ("1988", "settlement"): 2, ("2023", "people_ethnic"): 2}

    cli(["regions", "--workers", "2"])
    regions = pd.read_csv(PATHS["results"] / "visual" / "regions.csv", dtype={"edition": str})
    assert len(regions) == 2 and set(regions["page"]) == {1}                       # one illustration per book
    assert regions["caption_people"].all()                                        # 'Campesinos y obreros ...'
    cli(["regions", "--workers", "2"])                                            # incremental: nothing new
    assert len(pd.read_csv(PATHS["results"] / "visual" / "pages_visual.csv")) == 4
    cli(["visual-sample"])
    cli(["visual-report"])
    assert (PATHS["results"] / "visual" / "report.md").exists()
