"""Prepare the obstacle (pane) textures from the digital source images.

Run:  python scripts/make_pane_textures.py

Input : assets/textures/raw/pane_{human,car,building}.jpg  (colour originals)
Output: assets/textures/panes/pane_{human,car,building}.png

The obstacles are printed in black and white, so each image is converted to
greyscale. The background is pushed to paper white and the image is cropped
to the drawing plus a small white margin. The printed size is set in
configs/arena.yaml, not here; the script prints each texture's aspect ratio
so the width there can be kept consistent with the height.
"""

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "assets" / "textures" / "raw"
OUT = ROOT / "assets" / "textures" / "panes"
MARGIN = 0.04   # white border, as a fraction of the longer side
INK = 215       # greyscale values below this count as drawing


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("human", "car", "building"):
        grey = np.asarray(Image.open(RAW / f"pane_{name}.jpg").convert("L")).astype(np.float32)
        paper = np.median(grey[:8])          # top rows are background in all three
        grey = np.clip(grey * (255.0 / paper), 0, 255)
        ys, xs = np.nonzero(grey < INK)
        pad = int(MARGIN * max(xs.max() - xs.min(), ys.max() - ys.min()))
        h, w = grey.shape
        box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, w), min(ys.max() + pad, h))
        tex = Image.fromarray(grey.astype(np.uint8)).crop(box).convert("RGB")
        tex.save(OUT / f"pane_{name}.png")
        print(f"pane_{name}.png  {tex.width} x {tex.height} px  width/height = {tex.width / tex.height:.3f}")


if __name__ == "__main__":
    main()
