"""Render still images of the arena: robot camera, segmentation, top view, 3/4 view.

Run:  python scripts/render_preview.py [--config configs/arena_lab.yaml] [--seed 0] [--grid 12] [--shape K]

Output: scripts/out/preview_robot_cam.png    what the robot sees (yolo_width x yolo_height)
        scripts/out/preview_segmentation.png one colour per label (geom_classes)
        scripts/out/preview_top.png          the arena from above
        scripts/out/preview_overview.png     3/4 view, like Matthew's first renders
        scripts/out/preview_grid.png         robot camera over --grid random episodes
        scripts/out/preview_shapes.png       every arena shape from above

Every image comes from sim.randomize.EpisodeRandomizer, the same randomisation
used for training data.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sim  # noqa: F401,E402  selects the GL backend; must come before mujoco

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from sim.arena import geom_classes, load_config  # noqa: E402
from sim.randomize import ArenaPool  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
COLOURS = {"floor": (200, 200, 200), "carpet": (90, 90, 90), "box": (170, 120, 60),
           "wall": (120, 120, 200), "room": (40, 40, 40), "paper": (255, 255, 255),
           "human": (230, 40, 40), "car": (40, 160, 230), "building": (60, 200, 60),
           "robot": (0, 0, 255)}


def free_camera(lookat, distance, azimuth, elevation):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/arena_lab.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", type=int, default=12, help="episodes in the robot-camera grid")
    ap.add_argument("--shape", type=int, default=None, help="force arena shape K (0 = straight)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pool = ArenaPool(cfg)
    rng = np.random.default_rng(args.seed)
    if args.shape is None:
        k, model, data, rand = pool.sample(rng)
    else:
        k, (model, data, rand) = args.shape, pool.get(args.shape)
    info = rand.reset(data, rng)
    print(f"shape {k} ({info['layout']}), walls: {info['walls']}")
    x0, y0, x1, y1 = pool.layouts[k].bbox
    centre = [(x0 + x1) / 2, (y0 + y1) / 2, 0]
    OUT.mkdir(parents=True, exist_ok=True)

    cam = cfg["camera"]
    W, H = cam.get("yolo_width", cam["render_width"]), cam.get("yolo_height", cam["render_height"])
    with mujoco.Renderer(model, H, W) as r:
        r.update_scene(data, camera=cam["name"])
        Image.fromarray(r.render()).save(OUT / "preview_robot_cam.png")
        r.enable_segmentation_rendering()
        r.update_scene(data, camera=cam["name"])
        seg = r.render()[..., 0]
        labels = geom_classes(model)
        rgb = np.zeros((*seg.shape, 3), np.uint8)
        for g, lab in labels.items():
            rgb[seg == g] = COLOURS.get(lab, (255, 0, 255))
        Image.fromarray(rgb).save(OUT / "preview_segmentation.png")
        seen = sorted({labels[g] for g in np.unique(seg) if g >= 0})
        print("labels seen by the robot camera:", ", ".join(seen))

    with mujoco.Renderer(model, 720, 1280) as r:
        r.update_scene(data, camera=free_camera(centre, 4.6, 90, -89.9))
        Image.fromarray(r.render()).save(OUT / "preview_top.png")
        r.update_scene(data, camera=free_camera(centre, 4.2, 220, -35))
        Image.fromarray(r.render()).save(OUT / "preview_overview.png")
    tiles = []
    for _ in range(args.grid):
        if args.shape is None:
            _, m2, d2, r2 = pool.sample(rng)
        else:
            m2, d2, r2 = pool.get(args.shape)
        r2.reset(d2, rng)
        with mujoco.Renderer(m2, H, W) as r:
            r.update_scene(d2, camera=cam["name"])
            tiles.append(r.render())
    cols = 4
    rows = [np.concatenate(tiles[i:i + cols], axis=1) for i in range(0, len(tiles) - cols + 1, cols)]
    if rows:
        Image.fromarray(np.concatenate(rows, axis=0)).save(OUT / "preview_grid.png")
    shots = []
    for k2 in range(len(pool.layouts)):
        m2, d2, r2 = pool.get(k2)
        r2.reset(d2, rng)
        x0, y0, x1, y1 = pool.layouts[k2].bbox
        with mujoco.Renderer(m2, 360, 360) as r:
            r.update_scene(d2, camera=free_camera([(x0 + x1) / 2, (y0 + y1) / 2, 0], 5.5, 90, -89.9))
            shots.append(r.render())
    Image.fromarray(np.concatenate(shots, axis=1)).save(OUT / "preview_shapes.png")
    print(f"wrote 6 images to {OUT.relative_to(Path.cwd()) if OUT.is_relative_to(Path.cwd()) else OUT}")


if __name__ == "__main__":
    main()
