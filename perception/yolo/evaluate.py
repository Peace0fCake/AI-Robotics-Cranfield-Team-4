"""Evaluate the exported detector exactly as the robot runs it.

Run:  python perception/yolo/evaluate.py --model runs/detect/panes_v1/weights/best.onnx
                                         [--images datasets/panes/images/val] [--conf 0.25]

Same code path as on the Pi (deploy/pi/detector.py): the 480x192 band fed as
is, one class per box, class-wise NMS, a fixed confidence threshold. Works on
any folder of images with YOLO labels next door (images/... -> labels/...),
so the real test set is scored the same way once it is annotated.

Matching: a prediction and a ground-truth sheet are matched when their IoU is
at least --match-iou, best IoU first, whatever their classes; then
  same class          -> correct for that class
  different class     -> a class confusion
  unmatched truth     -> missed sheet
  unmatched prediction-> false alarm
Prints precision, recall and F1 per class and the confusion matrix (counts).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from deploy.pi.detector import CLASSES, PaneDetector, iou_matrix  # noqa: E402


def read_labels(path, w, h):
    """YOLO label file -> (classes, boxes x1 y1 x2 y2 in pixels)."""
    rows = [list(map(float, ln.split())) for ln in path.read_text().splitlines() if ln.strip()] \
        if path.exists() else []
    if not rows:
        return np.zeros(0, int), np.zeros((0, 4))
    a = np.array(rows)
    cx, cy, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    return a[:, 0].astype(int), np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)


def evaluate(det, images, match_iou):
    n = len(CLASSES)
    cm = np.zeros((n + 1, n + 1), int)   # rows: truth (last = none), cols: prediction (last = none)
    for img_path in images:
        band = np.asarray(Image.open(img_path).convert("RGB"))
        lab = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
        gt_cls, gt_box = read_labels(lab, band.shape[1], band.shape[0])
        dets = det(band)
        p_cls = np.array([d.cls for d in dets], int)
        p_box = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets]).reshape(-1, 4)
        ious = iou_matrix(gt_box, p_box)
        used_g, used_p = set(), set()
        for g, p in sorted(np.argwhere(ious >= match_iou), key=lambda gp: -ious[gp[0], gp[1]]):
            if g in used_g or p in used_p:
                continue
            used_g.add(g)
            used_p.add(p)
            cm[gt_cls[g], p_cls[p]] += 1
        for g in set(range(len(gt_cls))) - used_g:
            cm[gt_cls[g], n] += 1
        for p in set(range(len(p_cls))) - used_p:
            cm[n, p_cls[p]] += 1
    return cm


def report(cm):
    n = len(CLASSES)
    print(f"{'class':<12}{'P':>7}{'R':>7}{'F1':>7}{'truth':>8}")
    for c in range(n):
        tp = cm[c, c]
        p = tp / max(cm[:, c].sum(), 1)
        r = tp / max(cm[c, :].sum(), 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        print(f"{CLASSES[c]:<12}{p:7.3f}{r:7.3f}{f1:7.3f}{cm[c, :].sum():8d}")
    names = list(CLASSES) + ["(none)"]
    print("\nconfusion matrix (rows = truth, columns = prediction, counts)")
    print(" " * 12 + "".join(f"{s[:10]:>11}" for s in names))
    for i, s in enumerate(names):
        print(f"{s[:11]:<12}" + "".join(f"{v:>11d}" for v in cm[i]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", default="datasets/panes/images/val")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU")
    ap.add_argument("--match-iou", type=float, default=0.5)
    args = ap.parse_args()
    det = PaneDetector(REPO / args.model, conf=args.conf, iou=args.iou)
    images = sorted((REPO / args.images).glob("*.jpg")) + sorted((REPO / args.images).glob("*.png"))
    if not images:
        sys.exit(f"no images in {args.images}")
    print(f"{len(images)} images, conf {args.conf}, NMS IoU {args.iou}, match IoU {args.match_iou}\n")
    report(evaluate(det, images, args.match_iou))


if __name__ == "__main__":
    main()
