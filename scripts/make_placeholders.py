"""Generate placeholder textures into assets/placeholders/.

Run:  python scripts/make_placeholders.py

Every image carries a text label and an arrow in its top-left corner so that
flipped, mirrored or rotated texture mapping is obvious in a render.
Replace with real photos under assets/textures/ once they are taken.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "assets" / "placeholders"

# name -> (width px, height px, background RGB, label)
# Aspect ratios follow configs/arena.yaml (floor plaque 1.0 x 1.5 m, box side
# 0.5 x 0.3 m, pane A4 portrait).
SPECS = {
    "floor_0": (400, 600, (238, 238, 232), "FLOOR 0"),
    "floor_1": (400, 600, (232, 238, 238), "FLOOR 1"),
    "floor_2": (400, 600, (238, 232, 238), "FLOOR 2"),
    "box": (500, 300, (176, 137, 92), "BOX"),
    "wall": (600, 120, (150, 160, 170), "WALL"),
    "pane_human": (420, 594, (214, 72, 72), "HUMAN"),
    "pane_car": (420, 594, (66, 120, 214), "CAR"),
    "pane_building": (420, 594, (88, 168, 96), "BUILDING"),
    "background": (1024, 256, (120, 130, 110), "BACKGROUND"),
}


def make(name, w, h, colour, label):
    img = Image.new("RGB", (w, h), colour)
    d = ImageDraw.Draw(img)
    dark = tuple(int(c * 0.55) for c in colour)
    step = max(w, h) // 10
    for x in range(0, w, step):
        d.line([(x, 0), (x, h)], fill=dark, width=1)
    for y in range(0, h, step):
        d.line([(0, y), (w, y)], fill=dark, width=1)
    d.rectangle([0, 0, w - 1, h - 1], outline=dark, width=4)
    font = ImageFont.load_default(size=max(16, min(w, h) // 7))
    d.text((w // 2, h // 2), label, fill=(20, 20, 20), font=font, anchor="mm")
    # Orientation mark: "TL" plus an arrow pointing right along the top edge.
    small = ImageFont.load_default(size=max(12, min(w, h) // 12))
    d.text((8, 6), "TL >", fill=(20, 20, 20), font=small)
    img.save(OUT / f"{name}.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, spec in SPECS.items():
        make(name, *spec)
    print(f"wrote {len(SPECS)} textures to {OUT}")
