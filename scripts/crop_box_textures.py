"""Crop the cardboard-box face photos into textures.

Run:  python scripts/crop_box_textures.py

Input : assets/textures/raw/box_*.webp  (box photographed from above on grey carpet)
Output: assets/textures/boxes/box_*.png (face only), plus a printed size table.

The face is found by colour: cardboard is much redder than it is blue, the
carpet is not. Rows and columns that are mostly cardboard give the bounding
box, which ignores small warm-coloured intruders such as a shoe in the corner.
Rough by design: no perspective correction.

Box axes used below: A = longest edge, B = middle edge, C = shortest edge.
In the arena the boxes stand on an A x C face, so B is vertical.
"""

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "assets" / "textures" / "raw"
OUT = ROOT / "assets" / "textures" / "boxes"

# raw name -> (output name, rotation in degrees counter-clockwise, face)
# Rotations put every output in the orientation it has on a standing box:
# big and top faces A wide, end face C wide and B tall.
FACES = {
    "box_big_sticker": ("box_big_sticker", 90, "A x B"),
    "box_big_plain": ("box_big_plain", 0, "A x B"),
    "box_top": ("box_top", 90, "A x C"),
    "box_end": ("box_end", 90, "C x B"),
}

INSET = 0.015  # trim this fraction off each edge to drop carpet and shadow


def face_bounds(rgb):
    """Return (left, top, right, bottom) of the cardboard region."""
    r, b = rgb[..., 0].astype(int), rgb[..., 2].astype(int)
    mask = (r - b > 35) & (r > 120)
    cols, rows = mask.sum(axis=0), mask.sum(axis=1)
    xs = np.flatnonzero(cols > 0.5 * cols.max())
    ys = np.flatnonzero(rows > 0.5 * rows.max())
    return xs[0], ys[0], xs[-1] + 1, ys[-1] + 1


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for raw_name, (out_name, rot, face) in FACES.items():
        img = Image.open(RAW / f"{raw_name}.webp").convert("RGB")
        left, top, right, bottom = face_bounds(np.asarray(img))
        dx, dy = int((right - left) * INSET), int((bottom - top) * INSET)
        crop = img.crop((left + dx, top + dy, right - dx, bottom - dy))
        if rot:
            crop = crop.rotate(rot, expand=True)
        crop.save(OUT / f"{out_name}.png")
        w, h = crop.size
        print(f"{out_name:16s} {face}  {w:4d} x {h:4d} px  short/long = {min(w, h) / max(w, h):.3f}")


if __name__ == "__main__":
    main()
