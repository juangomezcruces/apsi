"""Illustration segmentation, captions and the zero-shot/report path on synthetic pages.
CLIP itself is replaced by a deterministic fake encoder: these tests check the plumbing,
not the model's judgement (that is what the hand-coded validation sample is for)."""
import shutil

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

from ltg import visual as V
from ltg.config import PATHS

from conftest import font, make_page

needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
LINES = ["Los pueblos originarios conservan sus lenguas.", "La comunidad decide en asamblea."]


def test_colorfulness():
    gray = np.full((50, 50, 3), 128, np.uint8)
    rgb = np.random.default_rng(0).integers(0, 255, (50, 50, 3)).astype(np.uint8)
    assert V.colorfulness(gray) == pytest.approx(0)
    assert V.colorfulness(rgb) > 50


def test_merge_boxes():
    merged = V._merge_boxes([(0, 0, 10, 10), (5, 5, 10, 10), (50, 50, 5, 5), (12, 12, 10, 10)])
    assert sorted(merged) == [(0, 0, 22, 22), (50, 50, 5, 5)]


@needs_tesseract
def test_segment_finds_illustration_not_text_or_panel():
    from ltg.extract import ocr_image
    img = make_page(LINES)
    d = ImageDraw.Draw(img)                       # flat colour panel holding only text: must be rejected
    d.rectangle([90, 300, 1180, 520], fill=(255, 230, 150))
    d.text((120, 380), "Recuerda: todos somos parte de la comunidad.", fill="black", font=font(30))
    rgb = np.asarray(img)
    words = ocr_image(img)
    boxes = V.segment(rgb, words)
    assert len(boxes) == 1
    x, y, w, h = boxes[0]
    assert abs(x - 200) < 40 and abs(y - 900) < 40 and abs(w - 800) < 60 and abs(h - 500) < 60
    cap = V.caption_for(boxes[0], words, rgb.shape[0])
    assert cap == "Campesinos y obreros en la plaza"
    assert V.PEOPLE_RE.search(V.fold(cap))
    assert not V.PEOPLE_RE.search("la plaza de armas")


@needs_tesseract
def test_text_only_page_has_no_regions():
    from ltg.extract import ocr_image
    img = make_page(LINES, illustration=False)
    assert V.segment(np.asarray(img), ocr_image(img)) == []


class FakeEncoder:
    """Text: hashed bag of words. Image: mean colour mapped to the 'crowd' or 'no_people' prompt."""
    logit_scale = 100.0

    def __init__(self):
        self.dim = 64

    def _vec(self, s):
        v = np.zeros(self.dim)
        for w in s.lower().split():
            v[hash(w) % self.dim] += 1
        return v / (np.linalg.norm(v) or 1)

    def texts(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def images(self, images):
        out = []
        for im in images:
            red = np.asarray(im)[..., 0].mean() > 128
            out.append(self._vec("a crowd of many people" if red else "a map or a diagram"))
        return np.stack(out)


def _fake_regions(n_books=3):
    rows, pages = [], []
    for ed in ("1988", "2023"):
        for b in range(n_books):
            code = f"P{b + 1}XXX"
            for k in range(4):
                crop = V.crops_dir() / ed / code / f"001_{k:02d}.jpg"
                crop.parent.mkdir(parents=True, exist_ok=True)
                red = (k % 2 == 0) if ed == "2023" else (k == 0)
                Image.new("RGB", (64, 64), (230, 20, 20) if red else (20, 20, 230)).save(crop)
                rows.append(dict(region_id=f"{ed}|{code}|1|{k}", edition=ed, code=code, page=1, k=k,
                                 area_frac=0.1, colorfulness=50.0, haar_faces=0, caption="",
                                 caption_people=bool(red), crop=str(crop.relative_to(PATHS["visual"]))))
            pages.append(dict(edition=ed, code=code, page=1, n_regions=4, illus_frac=0.4, page_colorfulness=40))
    V.out_dir().mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(V.out_dir() / "regions.csv", index=False)
    pd.DataFrame(pages).to_csv(V.out_dir() / "pages_visual.csv", index=False)


class FakeDetector:
    """Red crops have 8 people, blue ones none."""

    def __call__(self, images):
        return [8 if np.asarray(im)[..., 0].mean() > 128 else 0 for im in images]


def test_people_category():
    assert [V.people_category(n) for n in (0, 1, 2, 5, 6, 30)] == \
        ["no_people", "one_person", "small_group", "small_group", "crowd", "crowd"]


def test_zeroshot_sample_and_report(data_root):
    _fake_regions()
    s = V.illustration_sample(n_per_edition=100)
    assert len(s) == 24                                   # fewer than 100 per edition: all of them
    assert len(V.illustration_sample(n_per_edition=5)) == 10
    s = V.illustration_sample(n_per_edition=100)
    z = V.run_zeroshot(encoder=FakeEncoder())
    assert set(z["people"]) == {"crowd", "no_people"}
    assert {"people_p", "people:crowd", "medium", "activity"} <= set(z.columns)
    probs = z[[c for c in z.columns if c.startswith("people:")]].sum(axis=1)
    assert np.allclose(probs, 1, atol=1e-3)

    sheet = V.visual_sample(n_per_cell=2)
    assert len(sheet) == 8 and {"gold_people", "pred_people"} <= set(sheet.columns)
    html = (PATHS["results"] / "validation" / "visual_sample.html").read_text()
    assert html.count("<figure>") == 8 and "pred_" not in html          # coders do not see predictions

    tables = V.visual_report(B=50)
    people = tables["people"].set_index("measure")
    assert people.loc["with_people", "1988"].startswith("25.0")
    assert people.loc["with_people", "2023"].startswith("50.0")
    assert tables["source"]["source"].item() == "CLIP"
    assert (V.out_dir() / "report.md").exists()

    d = V.run_detector(detector=FakeDetector())
    assert set(d["people_det"]) == {"crowd", "no_people"} and d["n_persons"].max() == 8
    tables = V.visual_report(B=50)
    assert tables["source"]["source"].item() == "detector"
    assert tables["people"].set_index("measure").loc["crowd", "2023"].startswith("100.0")


def test_regions_restart_drops_unfinished_book(data_root):
    """A crash after a book's regions were appended but before its pages were: on restart
    those regions are dropped, so the book is redone without duplicates."""
    V.out_dir().mkdir(parents=True)
    pd.DataFrame([dict(edition="2023", code="A", page=1, n_regions=1, illus_frac=0.1, page_colorfulness=1)]).to_csv(
        V.out_dir() / "pages_visual.csv", index=False)
    pd.DataFrame([dict(region_id="2023|A|1|0", edition="2023", code="A"),
                  dict(region_id="2023|B|1|0", edition="2023", code="B")]).to_csv(V.out_dir() / "regions.csv", index=False)
    V.run_regions(workers=1)
    assert list(pd.read_csv(V.out_dir() / "regions.csv")["code"]) == ["A"]
