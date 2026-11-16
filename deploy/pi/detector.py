"""Sheet detector for the robot: YOLO exported to ONNX, run with ONNX Runtime.

Used on the Pi 5 (deploy/pi/bench_yolo.py, the control loop) and on the PC
(perception/yolo/evaluate.py), so both measure exactly the same pipeline.
Needs only numpy and onnxruntime: no PyTorch, no Ultralytics.

    det = PaneDetector("best.onnx")
    boxes = det(band)        # band: 192 x 480 x 3 uint8 RGB, the cropped camera band
    # -> list of Detection(cls, name, conf, x1, y1, x2, y2), pixel coordinates in the band

Export (on the PC):
    yolo export model=runs/detect/<run>/weights/best.pt format=onnx imgsz=192,480
The model then outputs (1, 4 + classes, anchors): box centre, size and one
score per class for every anchor. Duplicates are removed here with
class-wise non-maximum suppression (NMS). An NMS-free export
(nms=False, output (1, 300, 6)) is also accepted.
"""

from dataclasses import dataclass

import numpy as np

CLASSES = ("human", "car", "building", "sheet_back")


@dataclass
class Detection:
    cls: int
    name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


def iou_matrix(a, b):
    """IoU between every box of a (N, 4) and b (M, 4), boxes as x1, y1, x2, y2."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def nms(boxes, scores, iou_thr):
    """Greedy NMS: keep the best box, drop the ones overlapping it by more than iou_thr."""
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return keep


class PaneDetector:
    def __init__(self, onnx_path, conf=0.25, iou=0.5, threads=None):
        import onnxruntime as ort  # imported here so the module loads without it for docs/tests

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.h, self.w = inp.shape[2], inp.shape[3]
        self.conf, self.iou = conf, iou

    def preprocess(self, band):
        if band.shape[:2] != (self.h, self.w):
            raise ValueError(f"expected a {self.w}x{self.h} band, got {band.shape[1]}x{band.shape[0]}: "
                             "crop the camera frame to the YOLO band first")
        x = band.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        return np.ascontiguousarray(x)

    def postprocess(self, out):
        """out: the model output for ONE image, (4 + classes, anchors) or (300, 6)."""
        if out.shape[1] == 6:                             # NMS-free export: x1 y1 x2 y2 score cls
            keep = out[:, 4] >= self.conf
            return [Detection(int(c), CLASSES[int(c)], float(s), *map(float, b))
                    for b, s, c in zip(out[keep, :4], out[keep, 4], out[keep, 5])]
        pred = out.T                                      # (anchors, 4 + classes)
        scores_all = pred[:, 4:]
        cls = scores_all.argmax(axis=1)                   # one class per box: the most likely
        scores = scores_all[np.arange(len(cls)), cls]
        keep = scores >= self.conf
        if not keep.any():
            return []
        cx, cy, w, h = pred[keep, :4].T
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        boxes = np.clip(boxes, 0, [self.w, self.h, self.w, self.h])
        scores, cls = scores[keep], cls[keep]
        dets = []
        for c in np.unique(cls):                          # NMS within each class
            idx = np.where(cls == c)[0]
            for i in nms(boxes[idx], scores[idx], self.iou):
                j = idx[i]
                dets.append(Detection(int(c), CLASSES[int(c)], float(scores[j]), *map(float, boxes[j])))
        return sorted(dets, key=lambda d: -d.conf)

    def __call__(self, band):
        outputs = self.session.run(None, {self.input_name: self.preprocess(band)})
        return self.postprocess(outputs[0][0])          # first output, first (only) image
