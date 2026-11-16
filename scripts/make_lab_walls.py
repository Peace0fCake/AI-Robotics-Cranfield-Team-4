"""Build the lab wall textures from the photo shot facing the wall.

Run:  python scripts/make_lab_walls.py

Input : assets/textures/raw/lab/lab_wall_sockets_1.jpg   (3072 x 4080)
Output: assets/textures/background/lab_room_wall.png  room wall, 2.5 m tall, one tile
        assets/textures/background/lab_wall_tile.png  bottom 0.5 m of it ("flat" arena wall)
        scripts/out/lab_room_wall_preview.png         three tiles side by side (seam check)

Steps
  1. Level the photo: the trunking and skirting slope by about 1.9 deg.
  2. Crop from the top of the trunking down to the foot of the skirting,
     between the box in the bottom-left corner and the paper sheet on the right.
  3. Flatten the lighting along the wall (column brightness), so tiles match.
  4. Build one tile: a socket-free stretch of wall made tileable (cross-fade
     with a half-shifted copy; trunking, wall and skirting are horizontal, so
     the blend is invisible), repeated to TILE_M, with the double socket
     pasted back once.
  5. Extend the wall above the trunking with plain cream up to ROOM_H.

Scale: the UK double socket plate (146 mm) spans about 440 px after levelling,
so about 3.0 px/mm. The numbers below were read off this photo; a new photo
needs new numbers.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "assets" / "textures" / "raw" / "lab" / "lab_wall_sockets_1.jpg"
OUT = ROOT / "assets" / "textures" / "background"
PREVIEW = ROOT / "scripts" / "out"

TILT_DEG = 1.86                  # levels the trunking and skirting
PX_PER_M = 3010                  # source scale after levelling (socket plate 146 mm = 440 px)
TOP, BOTTOM = 936, 3480          # rows: top of trunking .. foot of skirting
PLAIN = (400, 1960)              # columns of a stretch with no socket
SOCKET = (2050, 2510)            # columns of the double socket (plate plus a margin)
SOCKET_ROWS = 340                # band rows holding the trunking; only these are pasted
TILE_M = 1.0                     # tile width, metres: one socket per metre (spacing unmeasured)
ROOM_H = 2.5                     # wall height, metres
ARENA_WALL_H = 0.5               # height of the "flat" arena wall, metres
OUT_PX_PER_M = 600               # output resolution


def flatten_columns(a, radius=200):
    """Divide each column by a smoothed column brightness, keep the mean."""
    col = a.mean(axis=(0, 2))
    kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
    smooth = np.convolve(np.pad(col, radius, mode="edge"), kernel, mode="valid")
    return a * (col.mean() / smooth)[None, :, None]


def tileable_x(a):
    """Cross-fade with a copy shifted by half the width: periodic along x."""
    shifted = np.roll(a, a.shape[1] // 2, axis=1)
    w = (1.0 - np.abs(np.linspace(-1.0, 1.0, a.shape[1])))[None, :, None]
    return w * a + (1.0 - w) * shifted


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    img = Image.open(SRC).convert("RGB").rotate(TILT_DEG, resample=Image.Resampling.BICUBIC,
                                                center=(1536, 2040))
    band = flatten_columns(np.asarray(img).astype(np.float32)[TOP:BOTTOM])

    plain = tileable_x(band[:, PLAIN[0]:PLAIN[1]])
    tile_px = round(TILE_M * PX_PER_M)
    reps = int(np.ceil(tile_px / plain.shape[1]))
    # Repeat the periodic stretch, then resample to exactly tile_px so the
    # joins stay on its periodic boundary (cutting it would leave a seam).
    reps_img = Image.fromarray(np.clip(np.concatenate([plain] * reps, axis=1), 0, 255).astype(np.uint8))
    tile = np.asarray(reps_img.resize((tile_px, plain.shape[0]), Image.Resampling.LANCZOS),
                      dtype=np.float32).copy()
    s0 = (tile_px - (SOCKET[1] - SOCKET[0])) // 2
    tile[:SOCKET_ROWS, s0:s0 + SOCKET[1] - SOCKET[0]] = band[:SOCKET_ROWS, SOCKET[0]:SOCKET[1]]

    # Plain cream above the trunking: the mean wall colour just below the
    # trunking, with fine noise so it does not look like flat paint in renders.
    cream = band[int(0.2 * band.shape[0]):int(0.8 * band.shape[0])].reshape(-1, 3).mean(axis=0)
    upper_px = round((ROOM_H - (BOTTOM - TOP) / PX_PER_M) * PX_PER_M)
    rng = np.random.default_rng(0)
    noise = Image.fromarray((rng.normal(128, 40, (upper_px, tile_px))).clip(0, 255).astype(np.uint8))
    noise = np.asarray(noise.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)[..., None]
    upper = cream[None, None, :] + (noise - 128) * 0.15
    wall = np.concatenate([upper, tile], axis=0)

    size = (round(TILE_M * OUT_PX_PER_M), round(ROOM_H * OUT_PX_PER_M))
    room = Image.fromarray(np.clip(wall, 0, 255).astype(np.uint8)).resize(size, Image.Resampling.LANCZOS)
    room.save(OUT / "lab_room_wall.png")
    print(f"wrote lab_room_wall.png  ({room.width} x {room.height}, {TILE_M} m x {ROOM_H} m)")

    low = room.crop((0, room.height - round(ARENA_WALL_H * OUT_PX_PER_M), room.width, room.height))
    low.save(OUT / "lab_wall_tile.png")
    print(f"wrote lab_wall_tile.png  ({low.width} x {low.height}, {TILE_M} m x {ARENA_WALL_H} m)")

    preview = Image.new("RGB", (3 * room.width, room.height))
    for i in range(3):
        preview.paste(room, (i * room.width, 0))
    preview.save(PREVIEW / "lab_room_wall_preview.png")
    print("wrote scripts/out/lab_room_wall_preview.png")


if __name__ == "__main__":
    main()
