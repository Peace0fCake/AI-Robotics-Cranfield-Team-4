"""Build a horizontally tileable wall texture from the trunking photo.

Run:  python scripts/make_wall_texture.py

Input : assets/textures/raw/wall_trunking.webp
Output: assets/textures/background/wall_trunking_tile.png   one period of the pattern
        assets/textures/background/window_strip.png         view through the window above it
        scripts/out/wall_tile_preview.png                   three tiles side by side (seam check)

One period = one double socket + one data panel + the blank gap after it.
The corner coordinates below were read off this specific photo; a new photo
needs new numbers.

Scale estimate: a UK double socket plate is 146 mm wide and measures about
135 px here, so 1 px is about 1.08 mm. That gives a period of about 0.38 m
and a wall height (floor to the front edge of the sill) of about 0.26 m.
Both are estimates until measured.
"""

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "assets" / "textures" / "raw" / "wall_trunking.webp"
OUT = ROOT / "assets" / "textures" / "background"
PREVIEW = ROOT / "scripts" / "out"

X0, PERIOD = 245, 350          # tile starts in the blank gap left of the first socket


def top(x):                     # front edge of the sill
    return 214 + 0.045 * (x - 300)


def bottom(x):                  # wall / floor junction
    return 458 + 0.033 * (x - 300)


WINDOW_BOX = (0, 0, 1024, 150)  # region above the window frame's lower edge


def match_edges(tile):
    """Linear brightness ramp so the right edge matches the left edge when tiled."""
    a = np.asarray(tile).astype(np.float32)
    w = a.shape[1]
    left = a[:, :6].mean(axis=(0, 1))
    right = a[:, -6:].mean(axis=(0, 1))
    mid = (left + right) / 2
    ramp = np.linspace(0.0, 1.0, w)[None, :, None]
    gain = (mid / left) * (1 - ramp) + (mid / right) * ramp
    return Image.fromarray(np.clip(a * gain, 0, 255).astype(np.uint8))


WALL_COLOUR = (226, 222, 212)   # cream; the photo is backlit by the window and reads dark brown


def relight(tile):
    """Scale each channel so the trunking surface (the median pixel) becomes WALL_COLOUR."""
    a = np.asarray(tile).astype(np.float32)
    surface = np.median(a.reshape(-1, 3), axis=0)
    return Image.fromarray(np.clip(a * (np.array(WALL_COLOUR) / surface), 0, 255).astype(np.uint8))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    img = Image.open(RAW).convert("RGB")

    x1 = X0 + PERIOD
    height = round(((bottom(X0) - top(X0)) + (bottom(x1) - top(x1))) / 2)
    # QUAD order: upper-left, lower-left, lower-right, upper-right
    quad = (X0, top(X0), X0, bottom(X0), x1, bottom(x1), x1, top(x1))
    tile = img.transform((PERIOD, height), Image.Transform.QUAD, quad, Image.Resampling.BICUBIC)
    tile = relight(match_edges(tile))
    tile.save(OUT / "wall_trunking_tile.png")

    img.crop(WINDOW_BOX).save(OUT / "window_strip.png")

    preview = Image.new("RGB", (PERIOD * 3, height))
    for i in range(3):
        preview.paste(tile, (PERIOD * i, 0))
    preview.save(PREVIEW / "wall_tile_preview.png")
    print(f"tile {PERIOD} x {height} px, about {PERIOD * 1.08 / 1000:.2f} x {height * 1.08 / 1000:.2f} m")


if __name__ == "__main__":
    main()
