"""Crawl the edition catalog pages and write `catalog.csv`, one row per book.

    python -m ltg catalog [--editions 1988,2023]

Book links are recognised by their viewer page name (`<CODE>.htm`), whether
they appear as <a href>, in an onclick, or inside a script, so the crawler does
not depend on the catalog's markup. PDF links next to a book are kept too.
"""
import argparse
import logging
import re
import time
import unicodedata
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import EDITIONS, HEADERS, PATHS, PERIOD

log = logging.getLogger("ltg.catalog")

# H<year> prefix only on the historical server; then level letter, grade digit,
# subject/version letters and digits. `(?!l)` keeps `primaria.html` out.
CODE_RE = re.compile(r"^(?:H(?P<year>\d{4}))?(?P<level>[A-Z])(?P<grade>\d)(?P<rest>[A-Z0-9]{2,10})$")
HTM_IN_TEXT_RE = re.compile(r"""(?P<url>[\w./:%-]*?(?P<code>(?:H\d{4})?[A-Z]\d[A-Z0-9]{2,10})\.htm)(?![a-zA-Z])""")
PDF_IN_TEXT_RE = re.compile(r"""(?P<url>[\w./:%-]+\.pdf)\b""", re.I)

LEVELS = {"P": "primaria", "S": "secundaria", "K": "preescolar", "T": "telesecundaria"}

# First match wins, so specific phrases come before the words they contain.
SUBJECTS = [
    ("proyectos comunitarios", "Proyectos comunitarios"),
    ("proyectos escolares", "Proyectos escolares"),
    ("proyectos de aula", "Proyectos de aula"),
    ("proyectos", "Proyectos"),
    ("etica, naturaleza", "Ética, naturaleza y sociedades"),
    ("de lo humano", "De lo humano y lo comunitario"),
    ("saberes", "Saberes y pensamiento científico"),
    ("multiples lenguajes", "Múltiples lenguajes"),
    ("historia", "Historia"),
    ("ciencias sociales", "Ciencias sociales"),
    ("formacion civica", "Formación cívica y ética"),
    ("civismo", "Formación cívica y ética"),
    ("geografia", "Geografía"),
    ("atlas", "Atlas"),
    ("conocimiento del medio", "Conocimiento del medio"),
    ("entidad donde vivo", "La entidad donde vivo"),
    ("ciencias naturales", "Ciencias naturales"),
    ("matematica", "Matemáticas"),
    ("desafios", "Matemáticas"),
    ("artes", "Artes"),
    ("artistica", "Artes"),
    ("vida saludable", "Vida saludable"),
    ("lengua materna", "Lengua"),
    ("espanol", "Lengua"),
    ("lecturas", "Lengua"),
    ("lenguajes", "Lengua"),
    ("trazos", "Lengua"),
    ("mi libro de", "Lengua"),
    ("ingles", "Inglés"),
]


def fold(s):
    """Lower-case and strip accents (for matching only, never for display)."""
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def subject_from_title(title):
    t = fold(title)
    for key, label in SUBJECTS:
        if key in t:
            return label
    return ""


def parse_code(code):
    m = CODE_RE.match(code or "")
    if not m:
        return None
    d = m.groupdict()
    return dict(code=code, hist_year=d["year"] or "", level=LEVELS.get(d["level"], d["level"]),
                grade=int(d["grade"]), code_rest=d["rest"])


def make_session(retries=4, backoff=2.0):
    s = requests.Session()
    s.headers.update(HEADERS)
    r = Retry(total=retries, backoff_factor=backoff, status_forcelist=(429, 500, 502, 503, 504),
              allowed_methods=("GET", "HEAD"), raise_on_status=False)
    s.mount("https://", HTTPAdapter(max_retries=r, pool_maxsize=8))
    s.mount("http://", HTTPAdapter(max_retries=r, pool_maxsize=8))
    return s


def fetch_html(session, url, timeout=30):
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    if not r.encoding or r.encoding.lower() == "iso-8859-1":
        r.encoding = r.apparent_encoding     # these pages are often served without a charset
    return r.text


def _clean(s):
    return re.sub(r"\s+", " ", s or "").strip()


def _anchor_title(a):
    parts = [a.get_text(" ", strip=True), a.get("title", "")]
    for img in a.find_all("img"):
        parts += [img.get("alt", ""), img.get("title", "")]
    title = _clean(" ".join(p for p in parts if p))
    if len(title) < 4:                      # bare cover image: use the enclosing card's text
        node = a.parent
        for _ in range(3):
            if node is None:
                break
            text = _clean(node.get_text(" ", strip=True))
            if 4 <= len(text) <= 200:
                return text
            node = node.parent
    return title


def extract_books(html, base_url):
    """Return {code: dict(viewer_url, title, pdf_url)} for every book linked from `html`."""
    soup = BeautifulSoup(html, "lxml")
    books = {}

    def add(code, url, title="", pdf=""):
        if not parse_code(code):
            return
        b = books.setdefault(code, dict(code=code, viewer_url=urljoin(base_url, url), title="", pdf_url=""))
        if title and len(title) > len(b["title"]):
            b["title"] = title
        if pdf and not b["pdf_url"]:
            b["pdf_url"] = urljoin(base_url, pdf)

    for a in soup.find_all("a"):
        href = a.get("href", "") + " " + a.get("onclick", "")
        for m in HTM_IN_TEXT_RE.finditer(href):
            add(m.group("code"), m.group("url"), _anchor_title(a))
        pdfs = PDF_IN_TEXT_RE.findall(href)
        if pdfs:
            # A PDF link belongs to the book whose code it contains, else to the card it sits in.
            for p in pdfs:
                code = next((c for c in re.findall(r"(?:H\d{4})?[A-Z]\d[A-Z0-9]{2,10}", p) if parse_code(c)), None)
                if code:
                    add(code, f"{code}.htm" if code not in books else books[code]["viewer_url"], pdf=p)

    # Codes that only appear in scripts or data attributes.
    for m in HTM_IN_TEXT_RE.finditer(html):
        if m.group("code") not in books:
            add(m.group("code"), m.group("url"))
    return books


def nav_links(html, base_url, follow_params):
    """Same-host links that keep the seed's values for `follow_params` (e.g. the year)."""
    base = urlparse(base_url)
    want = {k: parse_qs(base.query).get(k) for k in follow_params}
    out = set()
    for a in BeautifulSoup(html, "lxml").find_all("a", href=True):
        u = urlparse(urljoin(base_url, a["href"]))
        if u.netloc != base.netloc or not u.query:
            continue
        q = parse_qs(u.query)
        if all(q.get(k) == v for k, v in want.items()):
            out.add(urlunparse(u._replace(fragment="")))
    return sorted(out)


def expand_seed(seed, expand):
    if not expand:
        return [seed]
    u = urlparse(seed)
    urls = [seed]
    for key, values in expand.items():
        for v in values:
            q = parse_qs(u.query)
            q[key] = [v]
            urls.append(urlunparse(u._replace(query=urlencode(q, doseq=True))))
    return list(dict.fromkeys(urls))


def crawl_edition(spec, session, delay=1.0, fetch=None):
    fetch = fetch or (lambda url: fetch_html(session, url))
    queue = expand_seed(spec["seed"], spec.get("expand"))
    seen, books = set(), {}
    while queue:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            html = fetch(url)
        except Exception as e:                      # one bad listing page should not lose the rest
            log.warning("edition %s: %s failed: %s", spec["edition"], url, e)
            continue
        found = extract_books(html, url)
        log.info("edition %s: %s -> %d books", spec["edition"], url, len(found))
        for code, b in found.items():
            b.setdefault("found_on", url)
            books.setdefault(code, b)
            if b["title"] and len(b["title"]) > len(books[code]["title"]):
                books[code]["title"] = b["title"]
        if spec.get("follow_params"):
            queue += [u for u in nav_links(html, url, spec["follow_params"]) if u not in seen]
        if delay:
            time.sleep(delay)
    rows = []
    for code, b in sorted(books.items()):
        meta = parse_code(code)
        year_dir = re.search(r"/(\d{4})/[^/]+\.htm", b["viewer_url"])
        rows.append(dict(edition=spec["edition"], period=PERIOD.get(spec["edition"], ""), **meta,
                         served_year=year_dir.group(1) if year_dir else meta["hist_year"],
                         title=b["title"], subject=subject_from_title(b["title"]),
                         viewer_url=b["viewer_url"], pdf_url=b["pdf_url"], found_on=b.get("found_on", "")))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--editions", default="", help="comma-separated subset, default all")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--out", default=str(PATHS["catalog"]))
    args = ap.parse_args(argv)
    wanted = {e for e in args.editions.split(",") if e}
    session = make_session()
    rows = []
    for spec in EDITIONS:
        if wanted and spec["edition"] not in wanted:
            continue
        rows += crawl_edition(spec, session, delay=args.delay)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no books found: check the seeds in config.py and the network policy")
    PATHS["catalog"].parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    log.info("wrote %s: %d books\n%s", args.out, len(df),
             df.groupby(["edition", "level"]).size().to_string())
    blank = (df["subject"] == "").sum()
    if blank:
        log.info("%d books have no subject inferred from the title; fill them in by hand", blank)
    return df
