"""Measure how fast the sheet detector runs (on the Pi 5, or any machine).

Run:  python deploy/pi/bench_yolo.py --model best.onnx [--images DIR] [--n 300] [--threads 4]

Times the three stages the control loop pays for every frame, on 480x192
bands: preprocess (uint8 -> float tensor), inference (ONNX Runtime), and
postprocess (scores -> boxes, NMS). Prints mean, median and 95th percentile
per stage and the frame rate the total allows. The control loop runs at
10 Hz, so the whole pipeline (capture + YOLO + policy + serial to the Pico)
must stay well under 100 ms.

--images: a folder of 480x192 band images (for example a copy of
datasets/panes/images/val). Without it, random noise is used: the timing of
the network is the same, only postprocess changes with what is detected.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deploy.pi.detector import PaneDetector  # noqa: E402


def load_bands(folder, h, w, n):
    if folder is None:
        rng = np.random.default_rng(0)
        return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(min(n, 20))]
    from PIL import Image
    files = sorted(Path(folder).glob("*.jpg"))[:n] + sorted(Path(folder).glob("*.png"))[:n]
    if not files:
        sys.exit(f"no images in {folder}")
    return [np.asarray(Image.open(f).convert("RGB").resize((w, h))) for f in files]


def stats(ms):
    a = np.array(ms)
    return f"mean {a.mean():6.1f} ms   median {np.median(a):6.1f} ms   p95 {np.percentile(a, 95):6.1f} ms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", default=None)
    ap.add_argument("--n", type=int, default=300, help="timed frames")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--threads", type=int, default=None, help="ONNX Runtime threads (Pi 5: try 2, 3, 4)")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    det = PaneDetector(args.model, conf=args.conf, threads=args.threads)
    bands = load_bands(args.images, det.h, det.w, args.n)
    for i in range(args.warmup):                  # first runs allocate memory: not representative
        det(bands[i % len(bands)])

    pre, inf, post = [], [], []
    for i in range(args.n):
        band = bands[i % len(bands)]
        t0 = time.perf_counter()
        x = det.preprocess(band)
        t1 = time.perf_counter()
        out = det.session.run(None, {det.input_name: x})
        t2 = time.perf_counter()
        det.postprocess(out[0][0])
        t3 = time.perf_counter()
        pre.append((t1 - t0) * 1e3)
        inf.append((t2 - t1) * 1e3)
        post.append((t3 - t2) * 1e3)

    total = np.array(pre) + np.array(inf) + np.array(post)
    print(f"model {args.model}, input {det.w}x{det.h}, threads {args.threads or 'default'}, {args.n} frames")
    print(f"preprocess   {stats(pre)}")
    print(f"inference    {stats(inf)}")
    print(f"postprocess  {stats(post)}")
    print(f"total        {stats(total)}")
    print(f"-> {1000 / total.mean():.1f} frames/s on average, {1000 / np.percentile(total, 95):.1f} at p95")


if __name__ == "__main__":
    main()
