"""Straighten and crop the floor-plaque photos into textures.

Run:  python scripts/make_floor_textures.py

Input : assets/textures/raw/floor_plain.jpg, floor_blackline.jpg
        (1536 x 2040 working copies of the phone photos, shot from above)
Output: assets/textures/floor/floor_plain.png          plaque next to the box end
        assets/textures/floor/floor_plain_mirror.png   same, mirrored, for the middle plaque
        assets/textures/floor/floor_blackline.png      plaque with the black line
        all 800 x 1200 px = 1.0 m (arena x) by 1.5 m (arena y)

Orientation: image right = +x (towards the black line), image top = +y
(the trunking wall).

Steps per photo
  1. Perspective-correct the plaque's four corners to a rectangle. The corners
     were read off these specific photos; new photos need new numbers.
  2. Flatten the lighting: the photos contain a broad window reflection, which
     would otherwise sit at a fixed spot on the simulated floor. Dividing by a
     heavily blurred copy removes it and keeps seams, scuffs and the line.
     Glare is added back in the simulator, where it can be randomised.

The strip of floor hidden under the box wall at the bottom of each photo is
missing, so the textures are stretched by a few percent in y.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "assets" / "textures" / "raw"
OUT = ROOT / "assets" / "textures" / "floor"
SIZE = (800, 1200)

# name -> plaque corners in the raw image: upper-left, lower-left, lower-right, upper-right
CORNERS = {
    "floor_plain": ((338, 253), (302, 1598), (1256, 1600), (1216, 263)),
    "floor_blackline": ((252, 300), (152, 1580), (1130, 1578), (1132, 337)),
}


def flatten_lighting(img, blur_radius=80):
    a = np.asarray(img).astype(np.float32)
    lum = a.mean(axis=2)
    white = np.median(a.reshape(-1, 3), axis=0)
    # Hide dark marks (tape, scuffs) from the lighting estimate.
    filled = a.copy()
    filled[lum < 0.8 * np.median(lum)] = white
    light = Image.fromarray(filled.astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur_radius))
    light = np.asarray(light).astype(np.float32)
    return Image.fromarray(np.clip(a * (white / np.maximum(light, 1.0)), 0, 255).astype(np.uint8))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, corners in CORNERS.items():
        img = Image.open(RAW / f"{name}.jpg").convert("RGB")
        quad = tuple(v for point in corners for v in point)
        tex = img.transform(SIZE, Image.Transform.QUAD, quad, Image.Resampling.BICUBIC)
        tex = flatten_lighting(tex)
        tex.save(OUT / f"{name}.png")
        print(f"wrote {name}.png")
        if name == "floor_plain":
            ImageOps.mirror(tex).save(OUT / "floor_plain_mirror.png")
            print("wrote floor_plain_mirror.png")


if __name__ == "__main__":
    main()
