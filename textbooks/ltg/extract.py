"""Turn page images (or PDFs) into text, one JSON record per page.

    python -m ltg extract [--editions 1988] [--workers 4]

Output:
    text/<edition>/<CODE>.jsonl         edition, code, page, source, text, n_words, mean_conf
    boxes/<edition>/<CODE>/<nnn>.tsv.gz word boxes (left, top, width, height, conf, text, ...)

The word boxes are what the visual stage uses to tell text from illustration.
A PDF with a real text layer is used in preference to OCR; its pages are also
rendered to JPEG so the visual stage sees every book the same way.
"""
import argparse
import gzip
import io
import json
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import PATHS

os.environ.setdefault("OMP_THREAD_LIMIT", "1")     # one tesseract thread per worker process
log = logging.getLogger("ltg.extract")

BOX_COLS = ["block", "par", "line", "left", "top", "width", "height", "conf", "text"]
# The served pages are ~1000-1300 px tall. Measured on sample pages from 1988, 2022,
# 2023 and 2025, OCR at native size is as accurate as upscaling (and upscaling to
# 2200 px *lowered* confidence and invented tokens in illustrations), at half the
# time, so only images smaller than this are enlarged.
MIN_HEIGHT = 900


def dehyphenate(text):
    """Join words split across lines: 'pue-\\nblo. Los' -> 'pueblo.\\nLos'. Only when the
    continuation is lower-case, so 'auto-\\nCAD' and dashes before names are left alone."""
    return re.sub(r"(\w)[-­]\n([a-záéíóúñü]\S*) ?", lambda m: m.group(1) + m.group(2) + "\n", text)


def words_to_text(words):
    """Rebuild reading-order text from word boxes (block > paragraph > line)."""
    if words.empty:
        return ""
    paras = []
    for _, par in words.groupby(["block", "par"], sort=True):
        lines = [" ".join(w["text"].tolist()) for _, w in par.groupby("line", sort=True)]
        paras.append("\n".join(lines))
    return dehyphenate("\n\n".join(paras))


def ocr_image(img, lang="spa", psm=3):
    import pytesseract
    img = img.convert("L")
    scale = 1.0
    if img.height < MIN_HEIGHT:
        scale = MIN_HEIGHT / img.height
        img = img.resize((round(img.width * scale), MIN_HEIGHT), Image.LANCZOS)
    df = pytesseract.image_to_data(img, lang=lang, config=f"--psm {psm}",
                                   output_type=pytesseract.Output.DATAFRAME)
    df = df[(df["level"] == 5) & df["text"].notna()].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df = df.rename(columns={"block_num": "block", "par_num": "par", "line_num": "line"})
    for c in ("left", "top", "width", "height"):          # back to original image pixels
        df[c] = (df[c] / scale).round().astype(int)
    return df[BOX_COLS].reset_index(drop=True)


def _save_boxes(words, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        words.to_csv(f, sep="\t", index=False)


def load_boxes(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return pd.read_csv(f, sep="\t", keep_default_na=False, dtype={"text": str})


def _record(edition, code, page, source, words, size):
    text = words_to_text(words)
    conf = pd.to_numeric(words["conf"], errors="coerce")
    conf = conf[conf >= 0]
    return dict(edition=edition, code=code, page=page, source=source, text=text,
                n_words=int(len(words)), mean_conf=round(float(conf.mean()), 1) if len(conf) else None,
                width=size[0], height=size[1])


def ocr_page(args):
    edition, code, img_path, lang = args
    page = int(Path(img_path).stem)
    try:
        img = Image.open(img_path)
        words = ocr_image(img, lang=lang)
    except Exception as e:                                   # a corrupt page must not stop the book
        log.warning("%s/%s p%d: %s", edition, code, page, e)
        return dict(edition=edition, code=code, page=page, source="ocr", text="", n_words=0,
                    mean_conf=None, error=str(e))
    _save_boxes(words, PATHS["boxes"] / edition / code / f"{page:03d}.tsv.gz")
    return _record(edition, code, page, "ocr", words, img.size)


def pdf_pages(edition, code, pdf_path, dpi=150, min_chars=80):
    """Text-layer extraction. Returns None if the PDF is a scan without usable text."""
    import fitz
    doc = fitz.open(pdf_path)
    chars = sum(len(p.get_text("text")) for p in doc)
    if chars < min_chars * max(len(doc), 1) * 0.3:
        return None
    out, img_dir = [], PATHS["pages"] / edition / code
    img_dir.mkdir(parents=True, exist_ok=True)
    zoom = dpi / 72
    for i, p in enumerate(doc, start=1):
        pix = p.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img_path = img_dir / f"{i:03d}.jpg"
        if not img_path.exists():
            Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB").save(img_path, quality=90)
        rows = [dict(block=b, par=0, line=l, left=round(x0 * zoom), top=round(y0 * zoom),
                     width=round((x1 - x0) * zoom), height=round((y1 - y0) * zoom), conf=100, text=t)
                for x0, y0, x1, y1, t, b, l, _ in p.get_text("words")]
        words = pd.DataFrame(rows, columns=BOX_COLS)
        _save_boxes(words, PATHS["boxes"] / edition / code / f"{i:03d}.tsv.gz")
        out.append(_record(edition, code, i, "pdf", words, (pix.width, pix.height)))
    return out


def extract_book(edition, code, workers=4, lang="spa", pool=None, force=False):
    out_path = PATHS["text"] / edition / f"{code}.jsonl"
    img_dir = PATHS["pages"] / edition / code
    images = sorted(img_dir.glob("[0-9][0-9][0-9].jpg")) if img_dir.exists() else []
    if out_path.exists() and not force:
        n_done = sum(1 for _ in out_path.open(encoding="utf-8"))
        if n_done and n_done >= len(images):
            return n_done
    records = None
    pdf = PATHS["pdf"] / edition / f"{code}.pdf"
    if pdf.exists():
        try:
            records = pdf_pages(edition, code, pdf)
        except Exception as e:
            log.warning("%s/%s: PDF unreadable (%s); falling back to OCR", edition, code, e)
    if records is None:
        if not images:
            log.warning("%s/%s: nothing to extract", edition, code)
            return 0
        jobs = [(edition, code, str(p), lang) for p in images]
        if pool is not None:
            records = list(pool.map(ocr_page, jobs, chunksize=4))
        else:
            records = [ocr_page(j) for j in jobs]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in sorted(records, key=lambda r: r["page"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    confs = [r["mean_conf"] for r in records if r.get("mean_conf") is not None]
    log.info("%s/%s: %d pages (%s), mean OCR confidence %s", edition, code, len(records),
             records[0]["source"] if records else "-", round(float(np.mean(confs)), 1) if confs else "n/a")
    return len(records)


def _download_complete(book_dir):
    m = book_dir / "manifest.json"
    if not m.exists():
        return True                     # pages placed by hand or rendered from a PDF
    try:
        return bool(json.loads(m.read_text()).get("complete"))
    except ValueError:
        return False


def books_on_disk(complete_only=False):
    """(edition, code) of every book with page images or a PDF. complete_only skips books
    whose download is still running or failed, so extract can run alongside download."""
    seen = set()
    for root in (PATHS["pages"], PATHS["pdf"]):
        if not root.exists():
            continue
        for d in root.iterdir():
            for p in d.iterdir():
                code = p.stem if p.suffix == ".pdf" else p.name
                if p.suffix == ".pdf" or (p.is_dir() and (not complete_only or _download_complete(p))):
                    seen.add((d.name, code))
    return sorted(seen)


def load_pages(editions=None):
    """All extracted pages as a DataFrame."""
    rows = []
    for path in sorted(PATHS["text"].glob("*/*.jsonl")):
        if editions and path.parent.name not in editions:
            continue
        with path.open(encoding="utf-8") as f:
            rows += [json.loads(line) for line in f]
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--editions", default="")
    ap.add_argument("--codes", default="")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--lang", default="spa")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--include-incomplete", action="store_true",
                    help="also books whose download is unfinished (pilot runs with --max-pages)")
    args = ap.parse_args(argv)
    eds = set(filter(None, args.editions.split(",")))
    codes = set(filter(None, args.codes.split(",")))
    books = [(e, c) for e, c in books_on_disk(complete_only=not args.include_incomplete)
             if (not eds or e in eds) and (not codes or c in codes)]
    log.info("%d books to extract with %d workers", len(books), args.workers)
    with ProcessPoolExecutor(args.workers) as pool:
        for e, c in books:
            extract_book(e, c, args.workers, args.lang, pool, args.force)
