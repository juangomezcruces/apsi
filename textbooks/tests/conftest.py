"""Shared fixtures. Everything is synthetic and local: no test touches the network."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ltg import config  # noqa: E402

FONT_CANDIDATES = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                   "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                   "/Library/Fonts/Arial.ttf", "C:/Windows/Fonts/arial.ttf"]


def font(size):
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    pytest.skip("no TrueType font available for synthetic page images")


@pytest.fixture
def data_root(tmp_path):
    old = config.DATA
    config.set_data_root(tmp_path / "data")
    yield tmp_path / "data"
    config.set_data_root(old)


def make_page(lines, size=(1275, 1650), illustration=True, seed=0):
    """A white page with Spanish text at the top and a textured colour 'illustration' with a caption."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    f = font(34)
    y = 80
    for line in lines:
        d.text((90, y), line, fill="black", font=f)
        y += 52
    if illustration:
        x0, y0, x1, y1 = 200, 900, 1000, 1400
        noise = rng.integers(0, 255, size=(y1 - y0, x1 - x0, 3), dtype=np.uint8)
        img.paste(Image.fromarray(noise), (x0, y0))
        for _ in range(12):
            cx, cy = rng.integers(x0, x1), rng.integers(y0, y1)
            d.ellipse([cx - 40, cy - 40, cx + 40, cy + 40], fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
        d.text((x0, y1 + 20), "Campesinos y obreros en la plaza", fill="black", font=font(28))
    return img
