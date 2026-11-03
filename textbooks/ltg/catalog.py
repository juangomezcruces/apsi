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
# The historical archive lists its books in a script: ["TITLE","H1988P5CI124",1988,5],
BOOK_ARRAY_RE = re.compile(r'\[\s*"(?P<title>[^"]*)"\s*,\s*"(?P<code>(?:H\d{4})?[A-Z]\d[A-Z0-9]{2,10})"'
                           r'\s*,\s*(?P<gen>\d{4})\s*,\s*(?P<grade>\d)\s*\]')
GRADE_RE = re.compile(r"Grado:?\s*(\d)")
TEACHER_RE = re.compile(r"para (la )?maestra|para el maestro|para (el |la )?docente|libro (del|para el) maestro|sin recetas")

LEVELS = {"P": "primaria", "S": "secundaria", "K": "preescolar", "T": "telesecundaria"}

# First match wins, so specific phrases come before the words they contain.
SUBJECTS = [
    ("sin recetas", "Libro para el docente"),
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
    ("cartografia", "Atlas"),
    ("sociales", "Ciencias sociales"),
    ("atlas", "Atlas"),
    ("constitucion", "Formación cívica y ética"),
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
    ("mi libro de", "Integrado (1°-2°)"),
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


def audience_from_title(title):
    t = fold(title)
    return "teacher" if TEACHER_RE.search(t) and "alumn" not in t else "student"


def _anchor_grade(a):
    for node in (a, a.parent, getattr(a.parent, "parent", None)):
        if node is not None:
            m = GRADE_RE.search(node.get_text(" ", strip=True))
            if m:
                return int(m.group(1))
    return None


def _anchor_title(a):
    """Card title: bold text, else the cover's alt text, else the link text, else the card's text."""
    for tag in a.find_all(["b", "strong", "h3", "h4"]):
        t = _clean(tag.get_text(" ", strip=True))
        if len(t) >= 4:
            return t.rstrip(". ")
    for img in a.find_all("img"):
        t = _clean(img.get("alt", "") or img.get("title", ""))
        if len(t) >= 4:
            return t.rstrip(". ")
    title = _clean(" ".join(p for p in (a.get("title", ""), a.get_text(" ", strip=True)) if p))
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

    def add(code, url, title="", pdf="", grade=None, access="viewer", year=""):
        if not parse_code(code):
            return
        b = books.setdefault(code, dict(code=code, viewer_url=urljoin(base_url, url) if url else "", title="",
                                        pdf_url="", grades=set(), access=access, served_year=year))
        if title and len(title) > len(b["title"]):
            b["title"] = title
        if pdf and not b["pdf_url"]:
            b["pdf_url"] = urljoin(base_url, pdf)
        if grade:
            b["grades"].add(int(grade))

    for a in soup.find_all("a"):
        href = a.get("href", "") + " " + a.get("onclick", "")
        for m in HTM_IN_TEXT_RE.finditer(href):
            add(m.group("code"), m.group("url"), _anchor_title(a), grade=_anchor_grade(a))
        if "reader.html" in href and "clave=" in href:
            # 2026 onwards: an encrypted-PDF reader and no image viewer. Recorded, not downloaded.
            q = parse_qs(urlparse(urljoin(base_url, a.get("href", ""))).query)
            add(q["clave"][0], None, _anchor_title(a), grade=_anchor_grade(a), access="pdf-reader",
                year=q.get("ciclo", [""])[0])
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


def books_from_script(js, base_url, generation=None):
    """Books declared as ["TITLE","CODE",generation,grade] arrays (historical archive)."""
    books = {}
    for m in BOOK_ARRAY_RE.finditer(js):
        if generation and m.group("gen") != str(generation):
            continue
        code = m.group("code")
        if not parse_code(code):
            continue
        b = books.setdefault(code, dict(code=code, viewer_url=urljoin(base_url, f"{code}.htm"),
                                        title=_clean(m.group("title")), pdf_url="", grades=set(),
                                        access="viewer", served_year=m.group("gen")))
        b["grades"].add(int(m.group("grade")))
    return books


def same_host_scripts(html, base_url):
    host = urlparse(base_url).netloc
    out = []
    for tag in BeautifulSoup(html, "lxml").find_all("script", src=True):
        u = urljoin(base_url, tag["src"])
        if urlparse(u).netloc == host and not re.search(r"jquery|bootstrap|modernizr|gtag|webfont", u, re.I):
            out.append(u)
    return out


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
        for js_url in same_host_scripts(html, url):
            if js_url in seen:
                continue
            seen.add(js_url)
            try:
                found.update(books_from_script(fetch(js_url), url, spec.get("generation")))
            except Exception as e:
                log.warning("edition %s: script %s failed: %s", spec["edition"], js_url, e)
        log.info("edition %s: %s -> %d books", spec["edition"], url, len(found))
        for code, b in found.items():
            b.setdefault("found_on", url)
            known = books.setdefault(code, b)
            if b["title"] and len(b["title"]) > len(known["title"]):
                known["title"] = b["title"]
            known["grades"] |= b["grades"]
        if spec.get("follow_params"):
            queue += [u for u in nav_links(html, url, spec["follow_params"]) if u not in seen]
        if delay:
            time.sleep(delay)
    rows = []
    for code, b in sorted(books.items()):
        meta = parse_code(code)
        meta["code_grade"] = meta.pop("grade")
        if meta["level"] not in LEVELS.values():        # e.g. Z7822, a workbook listed on the primaria page
            meta["level"] = spec.get("level", "primaria")
        grades = sorted(b["grades"]) or [meta["code_grade"]]   # the catalog's own grade beats the code's digit
        year_dir = re.search(r"/(\d{4})/[^/]+\.htm", b["viewer_url"])
        subject = subject_from_title(b["title"])
        rows.append(dict(edition=spec["edition"], period=PERIOD.get(spec["edition"], ""), **meta,
                         grade=grades[0], grades=";".join(map(str, grades)),
                         served_year=year_dir.group(1) if year_dir else (b.get("served_year") or meta["hist_year"]),
                         title=b["title"], subject=subject,
                         # one state book per child: kept out of the national comparison, reported apart
                         corpus="state" if subject == "La entidad donde vivo" else "national",
                         audience=audience_from_title(b["title"]), access=b.get("access", "viewer"),
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
