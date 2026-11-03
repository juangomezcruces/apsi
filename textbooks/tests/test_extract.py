"""OCR and PDF extraction on synthetic pages (needs tesseract-ocr + tesseract-ocr-spa)."""
import json
import shutil

import pytest

from ltg import extract as E
from ltg.config import PATHS

from conftest import make_page

LINES = ["La soberanía nacional reside en el pue-",
         "blo. Los campesinos y los obreros lucharon",
         "contra la dictadura de Porfirio Díaz."]

needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")


def test_dehyphenate():
    assert E.dehyphenate("en el pue-\nblo. Los") == "en el pueblo.\nLos"
    assert E.dehyphenate("ciuda-\ndanos y") == "ciudadanos\ny"
    assert E.dehyphenate("auto-\nCAD") == "auto-\nCAD"


@needs_tesseract
def test_ocr_page_roundtrip(data_root):
    img_dir = PATHS["pages"] / "1988" / "H1988P4CS001"
    img_dir.mkdir(parents=True)
    make_page(LINES).save(img_dir / "007.jpg", quality=92)
    n = E.extract_book("1988", "H1988P4CS001")
    assert n == 1
    rec = json.loads((PATHS["text"] / "1988" / "H1988P4CS001.jsonl").read_text())
    assert rec["page"] == 7 and rec["source"] == "ocr"
    text = rec["text"].lower()
    assert "pueblo." in text                          # re-joined across the line break
    assert "campesinos" in text and "dictadura" in text
    assert rec["mean_conf"] > 70
    boxes = E.load_boxes(PATHS["boxes"] / "1988" / "H1988P4CS001" / "007.tsv.gz")
    assert {"left", "top", "width", "height", "conf", "text"} <= set(boxes.columns)
    assert boxes["top"].min() < 200                   # coordinates are in original image pixels
    assert len(E.load_pages()) == 1
    # resumable: a second call does not redo the book
    assert E.extract_book("1988", "H1988P4CS001") == 1


def test_pdf_text_layer(data_root):
    fitz = pytest.importorskip("fitz")
    pdf = PATHS["pdf"] / "2023" / "P5PCA.pdf"
    pdf.parent.mkdir(parents=True)
    doc = fitz.open()
    for body in ["Los pueblos originarios conservan sus lenguas y tradiciones. " * 3,
                 "La comunidad se organiza en asamblea para decidir. " * 3]:
        doc.new_page().insert_textbox(fitz.Rect(50, 50, 550, 750), body, fontsize=12)
    doc.save(pdf)
    assert E.extract_book("2023", "P5PCA") == 2
    pages = E.load_pages(["2023"])
    assert list(pages["source"]) == ["pdf", "pdf"]
    assert "pueblos originarios" in pages.iloc[0]["text"]
    assert (PATHS["pages"] / "2023" / "P5PCA" / "002.jpg").exists()   # rendered for the visual stage
    assert ("2023", "P5PCA") in E.books_on_disk()
