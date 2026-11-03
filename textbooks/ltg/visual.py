"""How the books picture 'the people'.

    python -m ltg regions                     # cut illustrations out of every page (full corpus)
    python -m ltg illustration-sample         # simple random sample per edition for the costly models
    python -m ltg detect                      # person detector (Faster R-CNN, COCO) on the sample
    python -m ltg zeroshot [--model ...]      # CLIP zero-shot codebook on the sample
    python -m ltg visual-sample               # hand-coding sheet drawn from the sample
    python -m ltg visual-report               # per-edition tables -> results/visual/report.md

Density, colour and captions are measured on every illustration. People counts
(detector) and activity / figure / setting / medium (CLIP) are measured on a
simple random sample of illustrations per edition, because both models take
seconds per image on a CPU; edition-level proportions from a simple random
sample are unbiased, and the intervals still resample books.

Regions: page pixels that differ from the page background, minus the OCR word
boxes, closed and split into connected components. Flat coloured panels that
only held text are dropped by their low texture. Each region keeps the text
printed just below it as its caption.

The zero-shot codebook is a measurement instrument like any other: its labels
are only reportable after `visual-sample` has been hand-coded and
`python -m ltg validate --kind visual` shows acceptable agreement. Skin tone or
'race' is deliberately not inferred from pixels; code ethnic representation by
hand from visible cultural markers and context instead.
"""
import html
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import PATHS, PERIOD
from .extract import load_boxes
from .lexicon import FAMILIES, PEOPLE_FAMILIES
from .text_analysis import _with_ci, edition_order, fold, md_table, rate_table, stratified_sample

log = logging.getLogger("ltg.visual")
def out_dir():
    return PATHS["results"] / "visual"


def crops_dir():
    return PATHS["visual"] / "crops"


WORK_HEIGHT = 1200            # segment at this height; boxes are scaled back to the original


# --------------------------------------------------------------------------- features
def colorfulness(rgb):
    """Hasler & Süsstrunk (2003) colourfulness, ~0 for greyscale, >100 very colourful."""
    r, g, b = (rgb[..., i].astype(float) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def background_colour(rgb, border=0.03):
    h, w = rgb.shape[:2]
    bh, bw = max(1, int(h * border)), max(1, int(w * border))
    strip = np.concatenate([rgb[:bh].reshape(-1, 3), rgb[-bh:].reshape(-1, 3),
                            rgb[:, :bw].reshape(-1, 3), rgb[:, -bw:].reshape(-1, 3)])
    return np.median(strip, axis=0)


def _merge_boxes(boxes):
    """Merge overlapping (x, y, w, h) boxes until none overlap."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out = []
        while boxes:
            x, y, w, h = boxes.pop()
            for o in out:
                ox, oy, ow, oh = o
                if x < ox + ow and ox < x + w and y < oy + oh and oy < y + h:
                    nx, ny = min(x, ox), min(y, oy)
                    o[:] = [nx, ny, max(x + w, ox + ow) - nx, max(y + h, oy + oh) - ny]
                    changed = True
                    break
            else:
                out.append([x, y, w, h])
        boxes = out
    return [tuple(b) for b in boxes]


def segment(rgb, words=None, min_area_frac=0.012, ink_thresh=40, min_texture=10.0):
    """Illustration boxes (x, y, w, h) in the coordinates of `rgb`."""
    import cv2
    h0, w0 = rgb.shape[:2]
    s = min(1.0, WORK_HEIGHT / h0)
    small = cv2.resize(rgb, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_AREA) if s < 1 else rgb.copy()
    h, w = small.shape[:2]
    bg = background_colour(small)
    ink = (np.abs(small.astype(int) - bg).max(axis=2) > ink_thresh).astype(np.uint8)
    text = np.zeros((h, w), bool)                  # where words were printed: not illustration
    if words is not None and len(words):
        conf = pd.to_numeric(words["conf"], errors="coerce").fillna(0)
        alpha = words["text"].astype(str).str.contains(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]{2,}", regex=True)
        pad = max(2, round(3 * s))
        for x, y, ww, hh in words.loc[(conf > 40) & alpha, ["left", "top", "width", "height"]].itertuples(index=False):
            text[max(0, round(y * s) - pad):round((y + hh) * s) + pad,
                 max(0, round(x * s) - pad):round((x + ww) * s) + pad] = True
    ink[text] = 0
    k = max(3, round(0.012 * w))
    mask = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = min_area_frac * h * w
    boxes = [tuple(stats[i, :4]) for i in range(1, n) if stats[i, 4] >= min_area * 0.5
             and stats[i, 2] * stats[i, 3] >= min_area]
    # Texture on non-text pixels away from the region's edge: a flat panel that only
    # held text scores ~0 once its words are masked; drawings and photos score high.
    lap = cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), cv2.CV_64F)
    free = ~cv2.dilate(text.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    keep = []
    for x, y, ww, hh in _merge_boxes(boxes):
        b = max(2, min(ww, hh) // 20)
        vals = lap[y + b:y + hh - b, x + b:x + ww - b][free[y + b:y + hh - b, x + b:x + ww - b]]
        if vals.size < 0.1 * ww * hh:              # almost all text
            continue
        if vals.std() >= min_texture and ww * hh < 0.97 * h * w:
            keep.append((round(x / s), round(y / s), round(ww / s), round(hh / s)))
    return sorted(keep, key=lambda b: (b[1], b[0]))


def caption_for(box, words, page_h, below=0.07):
    """Words printed inside the bottom of, or just below, a region."""
    if words is None or not len(words):
        return ""
    x, y, w, h = box
    cx = words["left"] + words["width"] / 2
    top = words["top"]
    sel = words[(cx > x) & (cx < x + w) & (top > y + 0.85 * h) & (top < y + h + below * page_h)]
    return " ".join(sel.sort_index()["text"].astype(str))     # OCR output is already in reading order


def _people_regex():
    stems = set()
    for fam in PEOPLE_FAMILIES:
        for t in FAMILIES[fam]["terms"]:
            stems.add(fold(t.split("/")[0]))
    alts = sorted((re.escape(s) for s in stems), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(alts) + r")(?:s|es)?\b")


PEOPLE_RE = _people_regex()


def haar_faces(rgb):
    """Frontal faces, OpenCV's stock Haar cascade. A weak baseline on drawings: report with care.
    None when this OpenCV build has no Haar cascades (OpenCV 5 dropped them; pin opencv<5)."""
    import cv2
    if not hasattr(haar_faces, "clf"):
        haar_faces.clf = None
        if hasattr(cv2, "CascadeClassifier"):
            haar_faces.clf = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        else:
            log.warning("OpenCV %s has no CascadeClassifier: haar_faces left empty", cv2.__version__)
    if haar_faces.clf is None:
        return None
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return len(haar_faces.clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)))


def regions_for_page(edition, code, img_path, crop_max=640):
    page = int(Path(img_path).stem)
    rgb = np.asarray(Image.open(img_path).convert("RGB"))
    H, W = rgb.shape[:2]
    box_path = PATHS["boxes"] / edition / code / f"{page:03d}.tsv.gz"
    words = load_boxes(box_path) if box_path.exists() else None
    rows = []
    for k, (x, y, w, h) in enumerate(segment(rgb, words)):
        crop = rgb[y:y + h, x:x + w]
        cap = caption_for((x, y, w, h), words, H)
        out = crops_dir() / edition / code / f"{page:03d}_{k:02d}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        im = Image.fromarray(crop)
        im.thumbnail((crop_max, crop_max))
        im.save(out, quality=88)
        rows.append(dict(region_id=f"{edition}|{code}|{page}|{k}", edition=edition, period=PERIOD.get(edition, ""),
                         code=code, page=page, k=k, x=x, y=y, w=w, h=h, area_frac=round(w * h / (W * H), 4),
                         colorfulness=round(colorfulness(crop), 1), haar_faces=haar_faces(crop),
                         caption=cap, caption_people=bool(PEOPLE_RE.search(fold(cap))),
                         crop=str(out.relative_to(PATHS["visual"]))))
    page_row = dict(edition=edition, code=code, page=page, n_regions=len(rows),
                    illus_frac=round(min(1.0, sum(r["area_frac"] for r in rows)), 4),
                    page_colorfulness=round(colorfulness(rgb[::4, ::4]), 1))
    return rows, page_row


def _regions_job(args):
    try:
        return regions_for_page(*args), None
    except Exception as e:                       # one bad page must not stop the corpus ...
        return ([], None), f"{args[2]}: {type(e).__name__}: {e}"


def run_regions(editions=None, workers=4):
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(p.parent.parent.name, p.parent.name, str(p))
            for p in sorted(PATHS["pages"].glob("*/*/[0-9][0-9][0-9].jpg"))
            if not editions or p.parent.parent.name in editions]
    regions, pages, errors = [], [], []
    with ProcessPoolExecutor(workers) as ex:
        for i, ((rows, page_row), err) in enumerate(ex.map(_regions_job, jobs, chunksize=8)):
            regions += rows
            if page_row:
                pages.append(page_row)
            if err:
                errors.append(err)
            if i and i % 2000 == 0:
                log.info("%d / %d pages", i, len(jobs))
    if errors:                                   # ... but a systematic failure must be loud
        log.warning("%d of %d pages failed, first: %s", len(errors), len(jobs), errors[0])
        if len(errors) == len(jobs):
            raise RuntimeError(f"every page failed; first error: {errors[0]}")
    out_dir().mkdir(parents=True, exist_ok=True)
    pd.DataFrame(regions).to_csv(out_dir() / "regions.csv", index=False)
    pd.DataFrame(pages).to_csv(out_dir() / "pages_visual.csv", index=False)
    log.info("%d illustrations on %d pages", len(regions), len(pages))


# --------------------------------------------------------------------------- zero-shot codebook
# English prompts: CLIP was trained on English captions. Several prompts per
# label are averaged. Edit freely, then re-validate.
CODEBOOK = {
    "people": {
        "no_people": ["an illustration with no people in it", "a picture of objects, animals or a landscape",
                      "a map or a diagram"],
        "one_person": ["a picture of one person", "a portrait of a single person"],
        "small_group": ["a picture of two to five people", "a small group of people"],
        "crowd": ["a crowd of many people", "a large group of people gathered together"],
    },
    "activity": {
        "protest": ["people protesting or marching with banners", "a demonstration in the street"],
        "assembly": ["people sitting in a community assembly", "a group of people discussing in a meeting"],
        "voting": ["people voting at a ballot box", "an election"],
        "collective_work": ["neighbours working together on a community task", "people building something together"],
        "farming": ["peasants working in the fields", "people harvesting crops"],
        "industrial_work": ["workers in a factory", "miners or industrial workers"],
        "armed_struggle": ["armed revolutionaries", "soldiers fighting in a historical battle"],
        "festival": ["people dancing at a traditional festival", "a community celebration"],
        "school": ["children in a classroom", "students learning at school"],
        "family_home": ["a family at home", "parents and children together in a house"],
        "everyday": ["people doing everyday activities", "people walking in a market"],
    },
    "figure": {
        "historical_portrait": ["a formal portrait of a historical figure", "a painted portrait of a national hero"],
        "ordinary_adults": ["ordinary adults in everyday clothes"],
        "traditional_dress": ["people wearing traditional indigenous clothing", "people in embroidered traditional dress"],
        "children": ["children playing", "a group of children"],
        "officials": ["politicians or officials in suits", "a government official giving a speech"],
        "soldiers": ["soldiers in uniform", "revolutionaries with rifles"],
    },
    "setting": {
        "village": ["a small rural village", "a town square with a church"],
        "city": ["a modern city street", "tall buildings in a city"],
        "fields": ["agricultural fields in the countryside", "a cornfield"],
        "industrial": ["a factory or industrial site"],
        "school": ["a school or classroom"],
        "home": ["the inside of a house"],
        "historical": ["a scene from colonial or ancient times", "pre-Hispanic temples or ancient ruins"],
        "nature": ["mountains, forests, rivers or the sea"],
    },
    "medium": {
        "photograph": ["a photograph"],
        "drawing": ["a drawing or cartoon illustration", "a children's book illustration"],
        "painting_mural": ["a mural painting", "a fine art painting"],
        "map_diagram": ["a map", "a diagram or chart"],
    },
}
CONDITIONAL_ON_PEOPLE = {"activity", "figure", "setting"}


class ClipEncoder:
    """Hugging Face CLIP. Needs the model weights (huggingface.co must be reachable once)."""

    def __init__(self, model="openai/clip-vit-large-patch14", device=None):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(model)
        self.logit_scale = float(self.model.logit_scale.detach().exp())

    def _norm(self, x):
        x = x.float().cpu().numpy()
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    # Tower -> pooled output -> projection, spelled out: get_*_features changed its
    # return type between transformers 4 and 5.
    def texts(self, texts):
        with self.torch.no_grad():
            t = self.proc(text=texts, return_tensors="pt", padding=True).to(self.device)
            pooled = self.model.text_model(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).pooler_output
            return self._norm(self.model.text_projection(pooled))

    def images(self, images):
        with self.torch.no_grad():
            t = self.proc(images=images, return_tensors="pt").to(self.device)
            pooled = self.model.vision_model(pixel_values=t["pixel_values"]).pooler_output
            return self._norm(self.model.visual_projection(pooled))


def label_embeddings(encoder, codebook=CODEBOOK, template="a picture from a school textbook: {}"):
    out = {}
    for dim, labels in codebook.items():
        names, vecs = [], []
        for label, prompts in labels.items():
            e = encoder.texts([template.format(p) for p in prompts]).mean(0)
            names.append(label)
            vecs.append(e / np.linalg.norm(e))
        out[dim] = (names, np.stack(vecs))
    return out


def classify(image_emb, label_emb, logit_scale):
    """Softmax over each dimension's labels: {dim: (labels, probs[n_images, n_labels])}."""
    out = {}
    for dim, (names, vecs) in label_emb.items():
        logits = logit_scale * image_emb @ vecs.T
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        out[dim] = (names, p / p.sum(1, keepdims=True))
    return out


def run_zeroshot(model="openai/clip-vit-base-patch16", batch=32, encoder=None, sample=True):
    regions = _analysis_regions(sample)
    encoder = encoder or ClipEncoder(model)
    lab = label_embeddings(encoder)
    rows = []
    for i in range(0, len(regions), batch):
        chunk = regions.iloc[i:i + batch]
        imgs = [Image.open(PATHS["visual"] / c).convert("RGB") for c in chunk["crop"]]
        res = classify(encoder.images(imgs), lab, encoder.logit_scale)
        for j, rid in enumerate(chunk["region_id"]):
            row = dict(region_id=rid)
            for dim, (names, p) in res.items():
                row[dim] = names[int(p[j].argmax())]
                row[f"{dim}_p"] = round(float(p[j].max()), 3)
                row.update({f"{dim}:{n}": round(float(v), 4) for n, v in zip(names, p[j])})
            rows.append(row)
        if i and i % (batch * 50) == 0:
            log.info("%d / %d crops", i, len(regions))
    z = pd.DataFrame(rows)
    z.to_csv(out_dir() / "zeroshot.csv", index=False)
    log.info("wrote %s (%d crops)", out_dir() / "zeroshot.csv", len(z))
    return z


# --------------------------------------------------------------------------- analysis sample
def _national(df):
    """Keep national-corpus books (drops the 2022 state books) using the catalog, if present."""
    if not PATHS["catalog"].exists():
        return df
    cat = pd.read_csv(PATHS["catalog"], dtype={"edition": str}, keep_default_na=False)
    if "corpus" not in cat:
        return df
    state = set(zip(cat.loc[cat["corpus"] == "state", "edition"], cat.loc[cat["corpus"] == "state", "code"]))
    return df[[k not in state for k in zip(df["edition"], df["code"])]]


def illustration_sample(n_per_edition=300, seed=0):
    """Simple random sample of national-corpus illustrations, n per edition."""
    regions = _national(pd.read_csv(out_dir() / "regions.csv", dtype={"edition": str}))
    parts = [g.sample(min(len(g), n_per_edition), random_state=seed) for _, g in regions.groupby("edition")]
    s = pd.concat(parts)
    s.to_csv(out_dir() / "regions_sample.csv", index=False)
    log.info("sampled %d illustrations: %s", len(s), s.groupby("edition").size().to_dict())
    return s


def _analysis_regions(sample=True):
    path = out_dir() / ("regions_sample.csv" if sample else "regions.csv")
    if sample and not path.exists():
        raise SystemExit("run `python -m ltg illustration-sample` first (or pass --all)")
    return pd.read_csv(path, dtype={"edition": str})


def people_category(n):
    return "no_people" if n == 0 else "one_person" if n == 1 else "small_group" if n <= 5 else "crowd"


class PersonDetector:
    """torchvision Faster R-CNN (ResNet-50 FPN v2, COCO). On 24 hand-checked NEM crops it got
    people presence right 20 times and the none/one/group/crowd category 19 times, against 16
    and 15 for CLIP B/16. Drawings are out of its training domain: validate."""

    def __init__(self, score=0.6, min_size=640, device=None):
        import torch
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
        self.torch, self.score = torch, score
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                                                min_size=min_size, max_size=1000).to(self.device).eval()

    def __call__(self, images):
        from torchvision.transforms.functional import to_tensor
        with self.torch.no_grad():
            out = self.model([to_tensor(im).to(self.device) for im in images])
        return [int(((o["labels"] == 1) & (o["scores"] > self.score)).sum()) for o in out]


def run_detector(sample=True, detector=None, batch=4):
    regions = _analysis_regions(sample)
    det = detector or PersonDetector()
    rows = []
    for i in range(0, len(regions), batch):
        chunk = regions.iloc[i:i + batch]
        counts = det([Image.open(PATHS["visual"] / c).convert("RGB") for c in chunk["crop"]])
        rows += [dict(region_id=r, n_persons=n, people_det=people_category(n)) for r, n in zip(chunk["region_id"], counts)]
        if i and i % (batch * 50) == 0:
            log.info("%d / %d crops", i, len(regions))
    d = pd.DataFrame(rows)
    d.to_csv(out_dir() / "people_detect.csv", index=False)
    log.info("wrote %s: %s", out_dir() / "people_detect.csv", d["people_det"].value_counts().to_dict())
    return d


# --------------------------------------------------------------------------- validation
def visual_sample(n_per_cell=30, seed=0):
    """Stratified (edition x predicted people label) sample of the analysis sample, shuffled,
    with an HTML contact sheet. The people prediction is the detector's when it has run."""
    df = _analysis_regions(sample=(out_dir() / "regions_sample.csv").exists())
    zs, dt = out_dir() / "zeroshot.csv", out_dir() / "people_detect.csv"
    if zs.exists():
        df = df.merge(pd.read_csv(zs), on="region_id")
    if dt.exists():
        df = df.merge(pd.read_csv(dt), on="region_id")
        df["people"] = df["people_det"]
    if "people" not in df:
        df["people"] = ""
    s = stratified_sample(df, ["edition", "people"], n_per_cell, seed)
    vdir = PATHS["results"] / "validation"
    vdir.mkdir(parents=True, exist_ok=True)
    cols = ["region_id", "edition", "code", "page", "crop", "caption"] + [d for d in CODEBOOK if d in s]
    sheet = s[cols].rename(columns={d: f"pred_{d}" for d in CODEBOOK})
    for d in CODEBOOK:
        sheet[f"gold_{d}"] = ""
    sheet["notes"] = ""
    sheet.to_csv(vdir / "visual_sample.csv", index=False)
    # Contact sheet: predictions hidden so coders are not anchored by them.
    rel = Path("../../visual")
    cells = "\n".join(
        f'<figure><img loading="lazy" src="{html.escape(str(rel / c))}"><figcaption><b>{html.escape(r)}</b><br>'
        f'{html.escape(str(cap) if isinstance(cap, str) else "")}</figcaption></figure>'
        for r, c, cap in zip(sheet["region_id"], sheet["crop"], sheet["caption"]))
    codes = "".join(f"<li><b>{d}</b>: {', '.join(CODEBOOK[d])}</li>" for d in CODEBOOK)
    (vdir / "visual_sample.html").write_text(f"""<!doctype html><meta charset="utf-8">
<title>Visual coding sheet</title>
<style>body{{font:14px system-ui;margin:16px}}main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}}
figure{{margin:0;border:1px solid #ccc;padding:6px}}img{{max-width:100%;max-height:260px;display:block;margin:auto}}</style>
<h1>Visual coding sheet ({len(sheet)} illustrations)</h1>
<p>Code each region in <code>visual_sample.csv</code> (gold_* columns). Labels:</p><ul>{codes}</ul>
<p>activity / figure / setting only when people are present; leave blank otherwise.</p>
<main>{cells}</main>""", encoding="utf-8")
    log.info("wrote %s and visual_sample.html (%d regions)", vdir / "visual_sample.csv", len(sheet))
    return sheet


# --------------------------------------------------------------------------- report
def visual_report(B=1000):
    regions = _national(pd.read_csv(out_dir() / "regions.csv", dtype={"edition": str}))
    pages = _national(pd.read_csv(out_dir() / "pages_visual.csv", dtype={"edition": str}))
    eds = edition_order(pages["edition"].unique())
    tables = {}
    pb = pages.groupby(["edition", "code"]).agg(n_pages=("page", "size"), n_regions=("n_regions", "sum"),
                                                illus=("illus_frac", "sum")).reset_index()
    t = rate_table(pb, ["n_regions", "illus"], "n_pages", 1, B)
    tables["density"] = _with_ci(t).pivot(index="measure", columns="edition", values="value").reindex(columns=eds).reset_index()
    tables["colour"] = regions.groupby("edition")["colorfulness"].describe()[["count", "mean", "50%"]].reindex(eds).round(1).reset_index()

    full = regions.groupby(["edition", "code"]).agg(n=("region_id", "size"), cap=("caption_people", "sum")).reset_index()
    tables["captions_all"] = _with_ci(rate_table(full, ["cap"], "n", 100, B), 1).pivot(
        index="measure", columns="edition", values="value").reindex(columns=eds).reset_index()

    zs, dt = out_dir() / "zeroshot.csv", out_dir() / "people_detect.csv"
    if zs.exists() or dt.exists():
        df = regions[["region_id", "edition", "code", "caption_people"]]
        if zs.exists():
            df = df.merge(pd.read_csv(zs), on="region_id")
        if dt.exists():                                     # the detector, when run, supplies the people count
            df = df.merge(pd.read_csv(dt), on="region_id")
            df["people_clip"] = df.get("people")
            df["people"] = df["people_det"]
        tables["source"] = pd.DataFrame([dict(measure="people count", source="detector" if dt.exists() else "CLIP",
                                              n_illustrations=len(df))])
        df["with_people"] = df["people"] != "no_people"
        rb = df.groupby(["edition", "code"]).agg(
            n=("region_id", "size"), with_people=("with_people", "sum"),
            crowd=("people", lambda s: (s == "crowd").sum()),
            group=("people", lambda s: (s == "small_group").sum()),
            one=("people", lambda s: (s == "one_person").sum()),
            caption_people=("caption_people", "sum")).reset_index()
        t = rate_table(rb, ["with_people"], "n", 100, B)
        t2 = rate_table(rb.assign(n_people=rb["with_people"]), ["crowd", "group", "one"], "n_people", 100, B)
        tables["people"] = _with_ci(pd.concat([t, t2]), 1).pivot(index="measure", columns="edition",
                                                               values="value").reindex(columns=eds).reset_index()
        withp = df[df["with_people"]]
        for dim in ("activity", "figure", "setting", "medium"):
            if dim not in df:
                continue
            src = withp if dim in CONDITIONAL_ON_PEOPLE else df
            tables[dim] = (100 * pd.crosstab(src[dim], src["edition"], normalize="columns")).round(1).reindex(columns=eds).reset_index()
        cap = (100 * df.groupby(["edition", "with_people"])["caption_people"].mean()).unstack()
        cap = cap.rename(columns={True: "people images: % captions naming the people",
                                  False: "other images: % captions naming the people"})
        tables["captions"] = cap.reindex(eds).round(1).reset_index()

    lines = ["# 'The people' in the free textbooks: visual report (national corpus)", "",
             "Density, colour and captions: every illustration. People, activity, figure, setting, medium: "
             "the simple random sample of illustrations per edition (`illustration-sample`). Model labels are "
             "unvalidated until `visual-sample` has been hand-coded; see README.", ""]
    for key, title in [("source", "Where the labels come from"),
                       ("density", "Illustrations per page and share of page area (95% CI, books resampled)"),
                       ("colour", "Colourfulness of illustrations (Hasler-Süsstrunk)"),
                       ("captions_all", "% of all illustrations whose caption names the people"),
                       ("people", "% of illustrations with people; among those, % crowd / small group / one person"),
                       ("activity", "Activity, % of illustrations with people"),
                       ("figure", "Who is shown, % of illustrations with people"),
                       ("setting", "Setting, % of illustrations with people"),
                       ("medium", "Medium, % of all illustrations"),
                       ("captions", "Captions that name the people")]:
        if key in tables:
            lines += [f"## {title}", "", md_table(tables[key]), ""]
    out_dir().mkdir(parents=True, exist_ok=True)
    (out_dir() / "report.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", out_dir() / "report.md")
    return tables
