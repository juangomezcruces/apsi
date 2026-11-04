"""Downloader against a local HTTP server laid out like the documented URL scheme."""
import functools
import io
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image

from ltg import download as D
from ltg.catalog import make_session
from ltg.config import PATHS


def _jpeg(n):
    buf = io.BytesIO()
    Image.new("RGB", (60, 80), (n * 7 % 255, 100, 150)).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def server(tmp_path):
    site = tmp_path / "site"
    # declared count, pages 001-012
    (site / "2023/c/P1AAA").mkdir(parents=True)
    (site / "2023/P1AAA.htm").write_text("<script>var anchoPagina = 800; var totalPaginas = 12;</script>")
    for n in range(1, 13):
        (site / f"2023/c/P1AAA/{n:03d}.jpg").write_bytes(_jpeg(n))
    # nothing declared, pages 000-036, page 20 missing upstream
    (site / "c/H1988P3CS001").mkdir(parents=True)
    (site / "H1988P3CS001.htm").write_text("<html>visor</html>")
    for n in range(0, 37):
        if n != 20:
            (site / f"c/H1988P3CS001/{n:03d}.jpg").write_bytes(_jpeg(n))
    # viewer page with no images at all
    (site / "2023/P2ZZZ.htm").write_text("<html></html>")
    # page 2 served with an image content type but broken bytes (as H2018P4HIA/036 on the live site)
    (site / "2023/c/P3BAD").mkdir(parents=True)
    (site / "2023/P3BAD.htm").write_text("<script>ag_pages = 5;</script>")
    for n in range(0, 5):
        (site / f"2023/c/P3BAD/{n:03d}.jpg").write_bytes(b"\x11\xec\xf0\xd3 not a jpeg" * 50 if n == 2 else _jpeg(n))
    # 2025 layout: same viewer, but magazine.js loads pages from c2025/ (as on the live site)
    (site / "2025/c2025/P5PCA").mkdir(parents=True)
    (site / "2025/P5PCA.htm").write_text("<script>\nag_pages = 4;\nag_clave = \"P5PCA\";\n</script>")
    (site / "2025/magazine.js").write_text("  \timg.attr('src', './c2025/' + ag_clave + '/' + ag_page +'.jpg');")
    for n in range(0, 4):
        (site / f"2025/c2025/P5PCA/{n:03d}.jpg").write_bytes(_jpeg(n))

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(site)))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_declared_page_counts():
    html = "var anchoPagina = 800; var totalPaginas = 12; numPages: '240', zoom=3"
    assert D.declared_page_counts(html) == [12, 240]
    assert D.declared_page_counts('ag_pages = 362;\nag_clave = "P5PCA";') == [362]     # live viewer


def test_image_folder_from_magazine_js(server, data_root):
    row = dict(code="P5PCA", edition="2025", viewer_url=f"{server}/2025/P5PCA.htm", pdf_url="")
    m = D.download_book(row, make_session(retries=0), D.Throttle(0))
    assert (m["folder"], m["first"], m["last"], m["complete"]) == ("c2025", 0, 3, True)
    assert D.image_folder(make_session(retries=0), f"{server}/2023/P1AAA.htm") == "c"   # no magazine.js


def test_declared_count_book(server, data_root):
    row = dict(code="P1AAA", edition="2023", viewer_url=f"{server}/2023/P1AAA.htm", pdf_url="")
    m = D.download_book(row, make_session(retries=0), D.Throttle(0))
    assert (m["first"], m["last"], m["missing"], m["complete"]) == (1, 12, [], True)
    files = sorted(p.name for p in (PATHS["pages"] / "2023/P1AAA").glob("*.jpg"))
    assert files == [f"{n:03d}.jpg" for n in range(1, 13)]
    assert json.loads((PATHS["pages"] / "2023/P1AAA/manifest.json").read_text())["complete"]


def test_probed_book_with_gap_and_resume(server, data_root):
    row = dict(code="H1988P3CS001", edition="1988", viewer_url=f"{server}/H1988P3CS001.htm", pdf_url="")
    s = make_session(retries=0)
    m = D.download_book(row, s, D.Throttle(0))
    # binary search assumes contiguous pages; the gap at 20 sits below the true end, so it is found
    assert (m["first"], m["last"]) == (0, 36)
    assert m["missing"] == [20] and m["failed"] == [] and m["complete"]      # a 404 gap is recorded, not retried forever
    # resume: existing files are skipped, nothing is re-fetched
    status = D.fetch_file(s, D.Throttle(0), f"{server}/c/H1988P3CS001/005.jpg",
                          PATHS["pages"] / "1988/H1988P3CS001/005.jpg", D.JPEG_MAGIC)
    assert status == "skipped"


def test_pilot_limit_and_no_images(server, data_root):
    s = make_session(retries=0)
    row = dict(code="H1988P3CS001", edition="1988", viewer_url=f"{server}/H1988P3CS001.htm", pdf_url="")
    m = D.download_book(row, s, D.Throttle(0), max_pages=5)
    assert m["n_pages"] == 5 and not m["complete"]
    row = dict(code="P2ZZZ", edition="2023", viewer_url=f"{server}/2023/P2ZZZ.htm", pdf_url="")
    assert D.download_book(row, s, D.Throttle(0))["error"] == "no page images"


def test_minimal_hand_made_catalog(server, data_root, tmp_path):
    cat = tmp_path / "catalog.csv"
    cat.write_text(f"edition,code,viewer_url\n2023,P1AAA,{server}/2023/P1AAA.htm\n")
    res = D.main(["--catalog", str(cat), "--delay", "0"])
    assert len(res) == 1 and res[0]["complete"]


def test_teacher_and_reader_only_books_are_skipped(server, data_root, tmp_path):
    cat = tmp_path / "catalog.csv"
    cat.write_text("edition,code,viewer_url,level,audience\n"
                   f"2023,P1AAA,{server}/2023/P1AAA.htm,primaria,student\n"
                   f"2023,P1LPM,{server}/2023/P1LPM.htm,primaria,teacher\n"
                   "current,P5PCA,,primaria,student\n")
    res = D.main(["--catalog", str(cat), "--delay", "0"])
    assert [m["code"] for m in res] == ["P1AAA"]


def test_standardise_jpeg():
    import io
    big = io.BytesIO()
    Image.new("RGB", (1617, 2126), (200, 100, 50)).save(big, "JPEG", quality=95)
    out, changed = D.standardise_jpeg(big.getvalue())
    assert changed and Image.open(io.BytesIO(out)).size == (round(1617 * 1276 / 2126), 1276)
    small = _jpeg(1)
    assert D.standardise_jpeg(small) == (small, False)          # ordinary pages are left byte-identical


def test_corrupt_upstream_page_is_recorded_as_missing(server, data_root, monkeypatch):
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    row = dict(code="P3BAD", edition="2023", viewer_url=f"{server}/2023/P3BAD.htm", pdf_url="")
    m = D.download_book(row, make_session(retries=0), D.Throttle(0))
    assert (m["first"], m["last"], m["missing"], m["failed"], m["complete"]) == (0, 4, [2], [], True)
