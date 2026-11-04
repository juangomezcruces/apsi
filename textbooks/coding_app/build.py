"""Build the hand-coding app (a claude.ai Artifact page) from the validation samples.

    python coding_app/build.py --out <dir>      # -> <dir>/index.html + <dir>/img/*.jpg

The page embeds the four item sets (sense, role and visual validation samples, and
the Claude-coded training seeds, of which a fixed random 100 form the seed check) and
serves the illustration crops as files beside it. Each coder's codes are saved in the artifact's own store, under
codes/<coder id>/<task>/<task>-<row>, and can be exported from the page as CSVs in
the same format as results/validation/*.csv (gold_* columns filled) and
ltg/seeds/pueblo_senses.csv.

Rows are keyed by their position in the CSV, so rebuild only from the same sample
files the codes were made on.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from ltg.config import ERA, PATHS  # noqa: E402

VALID = ROOT / "results" / "validation"
SEEDS = ROOT / "ltg" / "seeds" / "pueblo_senses.csv"
MAX_SIDE, QUALITY = 640, 80
N_CHECK, SEED = 100, 0          # random seed rows re-coded as a check of Claude's labels


def _table(df):
    return {"cols": list(df.columns), "rows": df.astype(str).values.tolist()}


def load():
    read = lambda p: pd.read_csv(p, keep_default_na=False, dtype=str)
    sense, role, visual, seed = (read(VALID / "sense_sample.csv"), read(VALID / "role_sample.csv"),
                                 read(VALID / "visual_sample.csv"), read(SEEDS))
    cat = read(PATHS["catalog"])
    books = {f"{r.edition}|{r.code}": [r.title, r.subject, r.viewer_url]
             for r in cat.drop_duplicates(["edition", "code"]).itertuples()}
    used = set()
    for df, col in ((sense, "sent_id"), (role, "sent_id"), (seed, "source")):
        used |= {"|".join(s.split("|")[:2]) for s in df[col]}
    used |= set(visual["edition"] + "|" + visual["code"])
    check = [int(i) for i in np.random.default_rng(SEED).permutation(len(seed))[:N_CHECK]]   # in coding order
    return dict(tasks=dict(sense=_table(sense), role=_table(role), visual=_table(visual), seed=_table(seed)),
                books={k: v for k, v in books.items() if k in used}, era=ERA, seed_check=check)


def write_images(visual, out):
    d = out / "img"
    d.mkdir(parents=True, exist_ok=True)
    i = visual["cols"].index("crop")
    total = 0
    for n, row in enumerate(visual["rows"]):
        im = Image.open(PATHS["visual"] / row[i]).convert("RGB")
        im.thumbnail((MAX_SIDE, MAX_SIDE))
        p = d / f"visual-{n:04d}.jpg"
        im.save(p, quality=QUALITY, optimize=True, progressive=True)
        total += p.stat().st_size
    return total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--no-images", action="store_true")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    data = load()
    html = (HERE / "template.html").read_text(encoding="utf-8")
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    assert html.count("/*__DATA__*/null") == 1
    (a.out / "index.html").write_text(html.replace("/*__DATA__*/null", blob), encoding="utf-8")
    size = 0 if a.no_images else write_images(data["tasks"]["visual"], a.out)
    n = {k: len(v["rows"]) for k, v in data["tasks"].items()}
    print(f"wrote {a.out / 'index.html'} ({len(blob) / 1e3:.0f} kB data), items {n}, images {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
