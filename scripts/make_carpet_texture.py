"""Build a seamless carpet texture for the lab floor outside the arena.

Run:  python scripts/make_carpet_texture.py

Input : assets/textures/raw/lab/<SOURCE>
Output: assets/textures/floor/carpet_lab.png      one tile, SIZE x SIZE px
        scripts/out/carpet_tile_preview.png       3 x 3 tiles (seam check)

Steps
  1. Crop a square patch of carpet from the photo.
  2. Flatten the lighting: divide by a heavily blurred copy, so a bright or dark
     corner of the photo does not repeat on every tile in the simulator.
  3. Make it tileable: blend the patch with a copy shifted by half a tile.
     The shifted copy wraps around without a seam at the tile edges, so it is
     used near the edges; the original is used near the centre, where the
     shifted copy has its own seams.

Source: the current photo is shot almost along the floor, so the patch comes
from the bottom rows (the carpet nearest the lens), where perspective and
blur are smallest. It is a stand-in: a photo taken straight down at the
carpet replaces it by changing SOURCE and CROP. The carpet is only seen at a
grazing angle and from more than about 1.5 m in the simulator, so its mean
colour and grain matter more than fine detail.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "assets" / "textures" / "raw" / "lab"
OUT = ROOT / "assets" / "textures" / "floor"
PREVIEW = ROOT / "scripts" / "out"

SOURCE = "lab_floor_low_4.jpg"   # 3072 x 4080 phone photo
CROP = (1136, 3280, 1936, 4080)  # left, top, right, bottom: 800 x 800 px at the bottom centre
SIZE = 512                       # output tile, px


def flatten_lighting(img, blur_radius=120):
    """Divide out the slow brightness changes, keep the mean colour."""
    a = np.asarray(img).astype(np.float32)
    light = np.asarray(img.filter(ImageFilter.GaussianBlur(blur_radius))).astype(np.float32)
    mean = a.reshape(-1, 3).mean(axis=0)
    return a * (mean / np.maximum(light, 1.0))


def make_tileable_axis(a, axis):
    """Make a periodic along one axis by cross-fading with a half-shifted copy.

    The shifted copy wraps without a seam at the edges but has one in the
    middle line; the weight of the original is 1 on that whole line and 0 on
    the edges, so neither seam shows.
    """
    n = a.shape[axis]
    shifted = np.roll(a, n // 2, axis=axis)
    ramp = 1.0 - np.abs(np.linspace(-1.0, 1.0, n))       # 0 at the edges, 1 in the middle
    w = ramp[:, None, None] if axis == 0 else ramp[None, :, None]
    out = w * a + (1.0 - w) * shifted
    # Blending two images averages their grain and lowers the contrast most
    # where w is 0.5; scale the deviation from the mean back up there.
    gain = 1.0 / np.sqrt(w**2 + (1.0 - w) ** 2)
    mean = out.reshape(-1, 3).mean(axis=0)
    return mean + (out - mean) * gain


def make_tileable(a):
    """Horizontal pass, then vertical pass. The vertical pass keeps the
    horizontal periodicity, because it only mixes rows that are already periodic."""
    return make_tileable_axis(make_tileable_axis(a, axis=1), axis=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)

    img = Image.open(RAW / SOURCE).convert("RGB").crop(CROP)
    img = img.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    a = make_tileable(flatten_lighting(img))
    tile = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    tile.save(OUT / "carpet_lab.png")
    print("wrote carpet_lab.png")

    preview = Image.new("RGB", (3 * SIZE, 3 * SIZE))
    for i in range(3):
        for j in range(3):
            preview.paste(tile, (i * SIZE, j * SIZE))
    preview.save(PREVIEW / "carpet_tile_preview.png")
    print("wrote scripts/out/carpet_tile_preview.png")


if __name__ == "__main__":
    main()
