"""Download the page images (and PDFs, when the catalog lists one) for each book.

    python -m ltg download [--editions 2023] [--codes P5PCA,P6PCA] [--max-pages 20]

Pages are `c/<CODE>/<nnn>.jpg` next to the viewer page. The page count is read
from the viewer page when it declares one and is always confirmed by probing the
image URLs, falling back to an exponential + binary search over page numbers.
Everything is resumable: existing, valid JPEGs are skipped.

Be polite: the defaults (2 workers, 0.5 s between requests per worker) keep the
load on the public server low. A full corpus is tens of thousands of requests.
"""
import argparse
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin

import pandas as pd

from .catalog import make_session
from .config import PATHS

log = logging.getLogger("ltg.download")

JPEG_MAGIC = b"\xff\xd8"
PDF_MAGIC = b"%PDF"
_BAD_NAME = re.compile(r"width|height|zoom|size|scale|margin|top|left|ancho|alto|altura|escala|tama|^w$|^h$", re.I)


def image_url(viewer_url, code, n, pad=3):
    return urljoin(viewer_url, f"c/{code}/{n:0{pad}d}.jpg")


class Throttle:
    """At most one request per `delay` seconds per thread."""

    def __init__(self, delay):
        self.delay, self.local = delay, threading.local()

    def wait(self):
        last = getattr(self.local, "last", 0.0)
        gap = self.delay - (time.monotonic() - last)
        if gap > 0:
            time.sleep(gap)
        self.local.last = time.monotonic()


def declared_page_counts(html):
    """Integers the viewer assigns to page-count-like JS names, most likely first."""
    out = {}
    for pos, m in enumerate(re.finditer(r"""([A-Za-z_$][\w$]*)\s*[:=]\s*['"]?(\d{1,4})\b""", html)):
        name, n = m.group(1), int(m.group(2))
        if _BAD_NAME.search(name) or not 2 <= n <= 1500:
            continue
        pag = re.search(r"pag|page", name, re.I) is not None
        tot = re.search(r"total|num|count|max|^n", name, re.I) is not None
        rank = 0 if pag and tot else 1 if pag else 2 if tot else None
        if rank is not None and n not in out:
            out[n] = (rank, pos)
    return sorted(out, key=out.get)


class Prober:
    def __init__(self, session, throttle, timeout=20):
        self.session, self.throttle, self.timeout = session, throttle, timeout
        self.cache = {}

    def exists(self, url):
        if url not in self.cache:
            self.throttle.wait()
            try:
                r = self.session.get(url, stream=True, timeout=self.timeout)
                head = next(r.iter_content(4), b"") if r.status_code == 200 else b""
                r.close()
                self.cache[url] = head.startswith(JPEG_MAGIC)
            except Exception as e:
                log.debug("probe %s: %s", url, e)
                self.cache[url] = False
        return self.cache[url]


def find_pages(prober, viewer_url, code, declared=(), cap=2000):
    """Return (first, last) page numbers, or None if no page image is reachable."""
    first = next((i for i in (0, 1, 2) if prober.exists(image_url(viewer_url, code, i))), None)
    if first is None:
        return None
    url = lambda n: image_url(viewer_url, code, n)
    for n in declared:                          # the viewer may count from 0 or from 1
        for last in (n + first - 1, n):
            if last > first and prober.exists(url(last)) and not prober.exists(url(last + 1)):
                return first, last
    lo, hi = first, first + 8                   # lo exists; grow hi until it does not
    while prober.exists(url(hi)):
        lo, hi = hi, min(hi * 2, cap)
        if lo == cap:
            return first, cap
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if prober.exists(url(mid)) else (lo, mid)
    return first, lo


def _valid(path, magic):
    try:
        with open(path, "rb") as f:
            return f.read(len(magic)) == magic and path.stat().st_size > 512
    except OSError:
        return False


def fetch_file(session, throttle, url, dest, magic, timeout=60, tries=3):
    if dest.exists() and _valid(dest, magic):
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(tries):
        throttle.wait()
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 200 and r.content.startswith(magic):
                tmp = dest.with_suffix(dest.suffix + ".part")
                tmp.write_bytes(r.content)
                tmp.replace(dest)
                return "ok"
            if r.status_code == 404:
                return "missing"
        except Exception as e:
            log.debug("%s attempt %d: %s", url, attempt + 1, e)
        time.sleep(2 ** attempt)
    return "failed"


def download_book(row, session, throttle, workers=2, max_pages=None, want_pdf=True):
    code, edition, viewer = row["code"], str(row["edition"]), row["viewer_url"]
    book_dir = PATHS["pages"] / edition / code
    manifest_path = book_dir / "manifest.json"
    if manifest_path.exists() and not max_pages:
        m = json.loads(manifest_path.read_text())
        if m.get("complete"):
            return m
    manifest = dict(code=code, edition=edition, viewer_url=viewer)

    if want_pdf and isinstance(row.get("pdf_url"), str) and row["pdf_url"].startswith("http"):
        dest = PATHS["pdf"] / edition / f"{code}.pdf"
        manifest["pdf"] = fetch_file(session, throttle, row["pdf_url"], dest, PDF_MAGIC, timeout=300)

    try:
        throttle.wait()
        html = session.get(viewer, timeout=30).text
        declared = declared_page_counts(html)
    except Exception as e:
        log.warning("%s: viewer page failed (%s); probing images directly", code, e)
        declared = []
    span = find_pages(Prober(session, throttle), viewer, code, declared)
    if span is None:
        log.warning("%s/%s: no page images found under %s", edition, code, image_url(viewer, code, 1))
        manifest.update(complete=False, error="no page images")
        book_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=1))
        return manifest
    first, last = span
    if max_pages:
        last = min(last, first + max_pages - 1)
    pages = list(range(first, last + 1))
    with ThreadPoolExecutor(workers) as ex:
        status = list(ex.map(lambda n: fetch_file(session, throttle, image_url(viewer, code, n),
                                                   book_dir / f"{n:03d}.jpg", JPEG_MAGIC), pages))
    bad = [n for n, s in zip(pages, status) if s in ("missing", "failed")]
    manifest.update(first=first, last=last, declared=declared[:3], n_pages=len(pages),
                    missing=bad, complete=not bad and not max_pages)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    log.info("%s/%s: pages %d-%d, %d new, %d missing", edition, code, first, last,
             status.count("ok"), len(bad))
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", default=str(PATHS["catalog"]))
    ap.add_argument("--editions", default="")
    ap.add_argument("--codes", default="", help="comma-separated book codes")
    ap.add_argument("--level", default="primaria", help="'' for every level in the catalog")
    ap.add_argument("--max-pages", type=int, default=None, help="pilot runs: first N pages per book")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--no-pdf", dest="pdf", action="store_false", default=True)
    args = ap.parse_args(argv)

    cat = pd.read_csv(args.catalog, dtype={"edition": str})
    if args.editions:
        cat = cat[cat["edition"].isin(args.editions.split(","))]
    if args.codes:
        cat = cat[cat["code"].isin(args.codes.split(","))]
    if args.level and "level" in cat:          # a hand-made catalog needs only edition, code, viewer_url
        cat = cat[cat["level"].isna() | (cat["level"] == args.level)]
    session, throttle = make_session(), Throttle(args.delay)
    results = [download_book(r, session, throttle, args.workers, args.max_pages, args.pdf)
               for r in cat.to_dict("records")]
    done = sum(bool(m.get("complete")) for m in results)
    log.info("%d/%d books complete", done, len(results))
    return results
