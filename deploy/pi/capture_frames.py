"""Capture real camera frames on the robot, set up exactly like the simulator.

Run on the Pi:  python deploy/pi/capture_frames.py --out eval/real/raw [--every 0.5] [--lens 1.0]

UNTESTED on the robot so far (written on the PC): check the first saved images.

Camera setup, matching configs/arena_lab.yaml and decision D-014:
  - sensor mode 2304x1296 (full field of view, binned)
  - central 4:3 crop of the sensor, full height (ScalerCrop), output 480x360
  - exposure, gain and white balance locked after a short auto warm-up, so the
    brightness does not drift between frames (guide section 9)
  - focus locked (LensPosition, in dioptres: 1.0 = sharpest at 1 m)

Saves, per frame:
  <out>/full/000123.png   the 480x360 frame
  <out>/band/000123.png   rows yolo_crop_rows (132-324): the 480x192 band YOLO sees
The band images are the ones to annotate for the real test set (YOLO format,
e.g. with Label Studio or Roboflow), then score with
  python perception/yolo/evaluate.py --model <best.onnx> --images datasets/real/images/test

Keys: Enter = save one frame now; with --every, frames are also saved every
N seconds. Ctrl+C to stop.
"""

import argparse
import select
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

SENSOR_MODE = (2304, 1296)     # binned full-FOV mode of the Camera Module 3 (IMX708)
OUT_SIZE = (480, 360)          # 4:3, as camera.yolo_width / yolo_height
BAND_ROWS = (132, 324)         # camera.yolo_crop_rows


def open_camera(lens_position, fps):
    from picamera2 import Picamera2  # only on the Pi
    from libcamera import controls

    cam = Picamera2()
    config = cam.create_video_configuration(
        main={"size": OUT_SIZE, "format": "RGB888"},
        sensor={"output_size": SENSOR_MODE},
        controls={"FrameRate": fps},
    )
    cam.configure(config)
    full_w, full_h = cam.camera_properties["PixelArraySize"]       # 4608 x 2592
    crop_w = full_h * 4 // 3                                         # 4:3, full height
    cam.set_controls({"ScalerCrop": ((full_w - crop_w) // 2, 0, crop_w, full_h)})
    cam.start()
    time.sleep(2.0)                                                  # let auto exposure / white balance settle
    md = cam.capture_metadata()
    cam.set_controls({
        "AeEnable": False, "AwbEnable": False,
        "ExposureTime": md["ExposureTime"], "AnalogueGain": md["AnalogueGain"],
        "ColourGains": md["ColourGains"],
        "AfMode": controls.AfModeEnum.Manual, "LensPosition": lens_position,
    })
    time.sleep(0.5)
    print(f"locked: exposure {md['ExposureTime']} us, gain {md['AnalogueGain']:.2f}, "
          f"colour gains {tuple(round(g, 2) for g in md['ColourGains'])}, lens {lens_position} dioptre")
    return cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/real/raw")
    ap.add_argument("--every", type=float, default=0.0, help="also save every N seconds (0 = Enter only)")
    ap.add_argument("--lens", type=float, default=1.0, help="LensPosition in dioptres")
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "full").mkdir(parents=True, exist_ok=True)
    (out / "band").mkdir(parents=True, exist_ok=True)
    i = len(list((out / "full").glob("*.png")))
    cam = open_camera(args.lens, args.fps)
    print("Enter = save a frame, Ctrl+C = stop")
    last = 0.0
    try:
        while True:
            frame = cam.capture_array()[..., ::-1]   # "RGB888" arrives in BGR byte order: flip to RGB
            now = time.monotonic()
            key = select.select([sys.stdin], [], [], 0)[0]
            if key:
                sys.stdin.readline()
            if key or (args.every and now - last >= args.every):
                last = now
                Image.fromarray(np.ascontiguousarray(frame)).save(out / "full" / f"{i:06d}.png")
                band = frame[BAND_ROWS[0]:BAND_ROWS[1]]
                Image.fromarray(np.ascontiguousarray(band)).save(out / "band" / f"{i:06d}.png")
                print(f"saved {i:06d}")
                i += 1
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
    print(f"{i} frames in {out}")


if __name__ == "__main__":
    main()
