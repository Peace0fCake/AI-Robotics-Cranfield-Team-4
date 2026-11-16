"""Generate a YOLO detection dataset from the randomised MuJoCo arena.

Run:  python perception/yolo/make_dataset.py [--n-train 6000] [--n-val 1000] [--seed 0]
                                             [--config configs/arena_lab.yaml] [--out datasets/panes]
      python perception/yolo/make_dataset.py --back-focus 3000   # add a sheet_back-rich split

Output (datasets/ is git-ignored):
  datasets/panes/images/{train,val}/000123.jpg   480x192 band of the robot camera
  datasets/panes/labels/{train,val}/000123.txt   one line per sheet: class xc yc w h (0-1)
  datasets/panes/data.yaml                       for Ultralytics (paths relative to this file)
  datasets/panes/manifest.csv                    shape, robot pose and box count per image
  scripts/out/dataset_preview.png                16 training images with their boxes drawn

Classes: 0 human, 1 car, 2 building, 3 sheet_back. sheet_back is a sheet whose
picture cannot be read: seen from behind (blank back) or nearly edge-on. It is
still an obstacle; the guide (Q22) treats an unreadable sheet like a human.

Per image
  1. Pick an arena shape and randomise an episode (sim/randomize.py).
  2. Put the robot at a random spot on any plaque, heading roughly along that
     plaque's direction of travel, so sheets are seen from near and far, not
     only from the start.
  3. Render RGB and segmentation at yolo_width x yolo_height, keep the
     yolo_crop_rows band.
  4. For each sheet, the box is the rectangle around its VISIBLE pixels
     (paper + picture) in the band. Sheets with too few visible pixels are
     dropped: a box the detector could never find only teaches it noise.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import sim  # noqa: F401,E402  selects the GL backend; must come before mujoco

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from sim.arena import load_config, set_robot  # noqa: E402
from sim.layout import CELL  # noqa: E402
from sim.randomize import ArenaPool  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CLASSES = ["human", "car", "building", "sheet_back"]
EDGE_ON_COS = 0.17          # |cos| below this (more than ~80 deg off) = picture unreadable
MIN_PIXELS = 40             # visible pixels for a sheet to be labelled
MIN_PICTURE_PIXELS = 25     # fewer visible picture pixels = class unreadable -> sheet_back
MIN_BOX = (4, 6)            # minimum box width, height in pixels
ROBOT_MARGIN = 0.15         # metres kept between the bumper and plaque edges / sheets
HEADING_SPREAD = 40.0       # robot heading = plaque heading +/- this, degrees
KEEP_EMPTY = 0.3            # share of sheet-free frames kept (they teach "nothing here")


def pane_geoms(model, cfg):
    """{geom_id: (pane index, is_picture)} for every paper and picture geom."""
    out = {}
    for j in range(cfg["panes"]["max_count"]):
        body = model.body(f"pane_{j}").id
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == body:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                out[g] = (j, "_image_" in name)
    return out


def random_robot_pose(layout, panes, rng, reverse_prob=0.0):
    """A bumper position on a random plaque, away from edges and sheets.
    With probability reverse_prob the robot faces back down the arena (sees sheet backs)."""
    for _ in range(200):
        p = layout.plaques[rng.integers(len(layout.plaques))]
        x = rng.uniform(p.x0 * CELL + ROBOT_MARGIN, p.x1 * CELL - ROBOT_MARGIN)
        y = rng.uniform(p.y0 * CELL + ROBOT_MARGIN, p.y1 * CELL - ROBOT_MARGIN)
        if all(np.hypot(x - q["x"], y - q["y"]) > ROBOT_MARGIN for q in panes):
            back = 180.0 if rng.random() < reverse_prob else 0.0
            return x, y, p.yaw_deg + back + rng.uniform(-HEADING_SPREAD, HEADING_SPREAD)
    return None


def sheet_labels(model, data, cfg, seg, panes, geoms, cam_id):
    """YOLO lines (class, xc, yc, w, h normalised) for the sheets seen in `seg`."""
    h, w = seg.shape[:2]
    ids = np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)
    cam_pos = data.cam_xpos[cam_id]
    lines = []
    for info in panes:
        j = info["pane"]
        mask = np.isin(ids, [g for g, (jj, _) in geoms.items() if jj == j])
        if mask.sum() < MIN_PIXELS:
            continue
        ys, xs = np.nonzero(mask)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        if x1 - x0 < MIN_BOX[0] or y1 - y0 < MIN_BOX[1]:
            continue
        # Front face normal = local +z of the pane body. Facing the camera, not
        # edge-on, and enough of the picture in view -> its class; else sheet_back.
        body = model.body(f"pane_{j}").id
        normal = data.xmat[body].reshape(3, 3)[:, 2]
        to_cam = cam_pos - data.xpos[body]
        cos = float(normal @ to_cam / np.linalg.norm(to_cam))
        two_sided = cfg["panes"].get("two_sided", True)
        picture = np.isin(ids, [g for g, (jj, pic) in geoms.items() if jj == j and pic]).sum()
        readable = (abs(cos) if two_sided else cos) > EDGE_ON_COS and picture >= MIN_PICTURE_PIXELS
        cls = CLASSES.index(info["class"]) if readable else CLASSES.index("sheet_back")
        lines.append(f"{cls} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                     f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    return lines


def generate(cfg, split, n, rng, out, writer, renderers, pool, back_focus=False):
    """back_focus: robot often turned round, and frames without a sheet_back are
    mostly skipped, to give the rare sheet_back class more examples."""
    cam = cfg["camera"]
    W, H = cam["yolo_width"], cam["yolo_height"]
    r0, r1 = cam["yolo_crop_rows"]
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    i = 0
    while i < n:
        k, model, data, rand = pool.sample(rng)
        info = rand.reset(data, rng)
        pose = random_robot_pose(pool.layouts[k], info["panes"], rng, 0.5 if back_focus else 0.0)
        if pose is None:
            continue
        set_robot(model, data, *pose)
        mujoco.mj_forward(model, data)
        if k not in renderers:
            renderers[k] = (mujoco.Renderer(model, H, W), pane_geoms(model, cfg),
                            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam["name"]))
        r, geoms, cam_id = renderers[k]
        r.disable_segmentation_rendering()
        r.update_scene(data, camera=cam["name"])
        rgb = r.render()[r0:r1]
        r.enable_segmentation_rendering()
        r.update_scene(data, camera=cam["name"])
        seg = r.render()[r0:r1]
        lines = sheet_labels(model, data, cfg, seg, info["panes"], geoms, cam_id)
        if not lines and rng.random() > KEEP_EMPTY:
            continue
        if back_focus and not any(ln.startswith("3 ") for ln in lines) and rng.random() > 0.25:
            continue
        name = f"{i:06d}"
        Image.fromarray(rgb).save(out / "images" / split / f"{name}.jpg", quality=95)
        (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        writer.writerow([split, name, info["layout"], *(round(v, 3) for v in pose), len(lines)])
        i += 1


def preview(out, path, n=16):
    """Draw the boxes of the first n training images on a 4-column grid."""
    colours = [(230, 40, 40), (40, 120, 230), (40, 170, 60), (240, 160, 0)]
    tiles = []
    for name in sorted((out / "images" / "train").glob("*.jpg"))[:n]:
        img = Image.open(name).convert("RGB")
        d = ImageDraw.Draw(img)
        for line in (out / "labels" / "train" / f"{name.stem}.txt").read_text().split("\n"):
            if not line:
                continue
            c, xc, yc, w, h = line.split()
            c, xc, yc, w, h = int(c), *(float(v) for v in (xc, yc, w, h))
            box = [(xc - w / 2) * img.width, (yc - h / 2) * img.height,
                   (xc + w / 2) * img.width, (yc + h / 2) * img.height]
            d.rectangle(box, outline=colours[c], width=2)
            d.text((box[0], max(box[1] - 11, 0)), CLASSES[c], fill=colours[c])
        tiles.append(np.asarray(img))
    rows = [np.concatenate(tiles[i:i + 4], axis=1) for i in range(0, len(tiles) - 3, 4)]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/arena_lab.yaml")
    ap.add_argument("--out", default="datasets/panes")
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--back-focus", type=int, default=0, metavar="N",
                    help="ONLY add N sheet_back-rich images as split train_back to an existing dataset")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = REPO / args.out
    pool = ArenaPool(cfg)
    renderers = {}
    out.mkdir(parents=True, exist_ok=True)
    if args.back_focus:
        # Edge-on sheets are rarer still: turn more panes far round for this split.
        cfg["panes"]["pose"]["yaw_wide_prob"] = 0.4
        with open(out / "manifest.csv", "a", newline="") as f:
            generate(cfg, "train_back", args.back_focus, np.random.default_rng(args.seed + 2_000_003), out,
                     csv.writer(f), renderers, pool, back_focus=True)
    else:
        with open(out / "manifest.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["split", "image", "shape", "robot_x", "robot_y", "robot_yaw_deg", "boxes"])
            # Separate random streams, so the validation set never repeats a training episode.
            generate(cfg, "train", args.n_train, np.random.default_rng(args.seed), out, writer, renderers, pool)
            generate(cfg, "val", args.n_val, np.random.default_rng(args.seed + 1_000_003), out, writer,
                     renderers, pool)
    for r, *_ in renderers.values():
        r.close()

    # No `path` key: Ultralytics then resolves train/val relative to this file.
    train_dirs = sorted(p.name for p in (out / "images").glob("train*") if p.is_dir())
    (out / "data.yaml").write_text(
        "# Generated by perception/yolo/make_dataset.py\n"
        "train: [" + ", ".join(f"images/{d}" for d in train_dirs) + "]\nval: images/val\n"
        "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    if args.back_focus:
        print(f"added {args.back_focus} train_back images to {out.relative_to(REPO)}; train = {train_dirs}")
    else:
        preview(out, REPO / "scripts" / "out" / "dataset_preview.png")
        print(f"wrote {args.n_train} train + {args.n_val} val images to {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
