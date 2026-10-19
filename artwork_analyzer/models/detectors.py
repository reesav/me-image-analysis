"""Pluggable detector backends.

Backends implement :func:`detect` returning (list[RawDetection], DetectionBackend).
The call sites are model-agnostic so review/export/store layers work the same
for any backend. Two real backends ship:

- :class:`YuNetBackend`: OpenCV ``FaceDetectorYN`` with the bundled YuNet ONNX
  (faces). Confidence is normalised from the raw YuNet quality score to 0-1
  via ``s / (1 + s)``.
- :class:`ORTSSDv1Backend`: ONNX Runtime SSD-MobileNet-v1 (VOC 20). Person is
  class 15 in VOC. Supports returning only a subset of labels.
- :class:`ORTYOLOv8Backend`: ONNX Runtime YOLOv8n (COCO). Optional alternative.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .registry import resolve_model

logger = logging.getLogger("artwork_analyzer.models.detectors")

COCO_PERSON_LABEL = "person"


def _int_dim(shape, idx, default):
    """Return a concrete int for a (possibly dynamic) ONNX shape dimension."""
    try:
        return int(shape[idx])
    except (TypeError, ValueError):
        return default


@dataclass
class RawDetection:
    label: str
    confidence: float  # 0-1
    x: float  # normalised
    y: float
    width: float
    height: float


@dataclass
class DetectionBackend:
    name: str
    model_identifier: str
    available: bool = True


# ---- VOC 20 labels for SSD-MobileNet-v1 (1-indexed classes; 0 = background) ----
VOC_LABELS = {
    1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle",
    6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow",
    11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike",
    15: "person", 16: "pottedplant", 17: "sheep", 18: "sofa",
    19: "train", 20: "tv",
}

# COCO 80 labels for YOLOv8 (0-indexed; 0 = person).
COCO_LABELS = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "kite",
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
    44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
    49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
    64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
    68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator",
    73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}


class YuNetBackend:
    """Face detection via OpenCV FaceDetectorYN + bundled YuNet ONNX."""

    MODEL_NAME = "face_yunet"

    def __init__(self, conf_threshold: float = 0.3, score_threshold: float = 0.25, nms_threshold: float = 0.3):
        self.conf_threshold = conf_threshold
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self._model_path = resolve_model(self.MODEL_NAME)
        self._net = None
        if self._model_path is None:
            logger.warning("YuNet model not available; face detection disabled.")

    @property
    def available(self) -> bool:
        return self._model_path is not None and self._model_path.exists()

    def detect(self, image_rgb: np.ndarray) -> Tuple[List[RawDetection], DetectionBackend]:
        info = DetectionBackend(name="yunet", model_identifier=self.MODEL_NAME, available=self.available)
        if not self.available:
            return [], info
        if self._net is None:
            self._net = cv2.FaceDetectorYN_create(
                str(self._model_path), "", (640, 640),
                score_threshold=self.score_threshold,
                nms_threshold=self.nms_threshold,
            )
        h, w = image_rgb.shape[:2]
        # FaceDetectorYN expects BGR input.
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        self._net.setInputSize((w, h))
        rc, dets = self._net.detect(bgr)
        results: List[RawDetection] = []
        if rc != 1:
            logger.warning("YuNet detect returned rc=%s", rc)
            return results, info
        if dets is None:
            return results, info
        for row in dets:
            x, y, bw, bh, raw_conf = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
            conf = float(np.clip(raw_conf / (1.0 + raw_conf), 0.0, 1.0))
            if conf < self.conf_threshold:
                continue
            results.append(RawDetection(
                label="face",
                confidence=conf,
                x=x / w, y=y / h, width=bw / w, height=bh / h,
            ))
        return results, info


class ORTCVBackend:
    """Base ONNX Runtime backend for ONNX object detectors."""

    def __init__(self, model_name: str, conf_threshold: float = 0.35):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self._session = None
        self._model_path = resolve_model(model_name)
        self._available = self._model_path is not None and self._model_path.exists()

    @property
    def available(self) -> bool:
        return self._available

    def _load(self):
        import onnxruntime as ort  # lazy
        if self._session is None:
            self._session = ort.InferenceSession(
                str(self._model_path), providers=["CPUExecutionProvider"]
            )
        return self._session


class ORTSSDv1Backend(ORTCVBackend):
    """SSD-MobileNet-v1 (VOC 20) via ONNX Runtime. ``person``=class 15.

    Outputs are normalised boxes (ymin,xmin,ymax,xmax) in 0-1, class index
    (1-20) and confidence.
    """

    OUTPUT_NAMES = ("detection_boxes:0", "detection_classes:0",
                    "detection_scores:0", "num_detections:0")

    def detect(self, image_rgb: np.ndarray,
               wanted_labels: Optional[Sequence[str]] = None) -> Tuple[List[RawDetection], DetectionBackend]:
        info = DetectionBackend(name="ssd_mobilenet_v1", model_identifier=self.model_name,
                                available=self.available)
        if not self.available:
            return [], info
        sess = self._load()
        inp = sess.get_inputs()[0]
        sz = _int_dim(inp.shape, 2, 300)
        resized = cv2.resize(image_rgb, (sz, sz))
        if inp.type.startswith("tensor(uint8)"):
            blob = resized.astype(np.uint8)[None, :, :, :]
        else:
            blob = (resized.astype(np.float32) / 255.0)[None, :, :, :]
        outputs = sess.run(None, {inp.name: blob})
        out_map = {o.name: outputs[i] for i, o in enumerate(sess.get_outputs())}

        def _get(name):
            arr = out_map.get(name)
            return np.atleast_1d(np.asarray(arr).ravel())

        boxes = np.atleast_2d(np.asarray(out_map[self.OUTPUT_NAMES[0]].reshape(-1, 4)))
        classes = _get(self.OUTPUT_NAMES[1])
        scores = _get(self.OUTPUT_NAMES[2])
        if self.OUTPUT_NAMES[3] in out_map:
            ndet = int(_get(self.OUTPUT_NAMES[3])[0])
        else:
            ndet = len(scores)
        results: List[RawDetection] = []
        for i in range(ndet):
            conf = float(scores[i])
            if conf < self.conf_threshold:
                continue
            cid = int(classes[i])
            label = VOC_LABELS.get(cid, str(cid))
            if wanted_labels is not None and label not in wanted_labels:
                continue
            if i >= len(boxes):
                continue
            y1, x1, y2, x2 = boxes[i].tolist()
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            results.append(RawDetection(label=label, confidence=conf,
                                        x=float(x1), y=float(y1),
                                        width=float(bw), height=float(bh)))
        return results, info


class ORTYOLOv8Backend(ORTCVBackend):
    """YOLOv8n (COCO) via ONNX Runtime. ``person``=class 0.

    Output layout from a standard Ultralytics export is ``[1, 85, N]`` (or
    ``[1, 84, N]``). The parser normalises to `[N, 85]` rows of
    (cx, cy, w, h, obj, 80 class scores), then applies a configurable
    objectness-class-score product threshold.
    """

    def detect(self, image_rgb: np.ndarray,
               wanted_labels: Optional[Sequence[str]] = None,
               conf_threshold: Optional[float] = None) -> Tuple[List[RawDetection], DetectionBackend]:
        thr = conf_threshold or self.conf_threshold
        info = DetectionBackend(name="yolov8n", model_identifier=self.model_name,
                                available=self.available)
        if not self.available:
            return [], info
        sess = self._load()
        h, w = image_rgb.shape[:2]
        inp = sess.get_inputs()[0]
        sz = _int_dim(inp.shape, 2, 640)
        resized = cv2.resize(image_rgb, (sz, sz))
        if inp.type.startswith("tensor(uint8)"):
            blob = resized.astype(np.uint8)[None, :, :, :]
        else:
            blob = (resized.astype(np.float32) / 255.0)[None, :, :, :]
        outputs = sess.run(None, {inp.name: blob})
        arr = outputs[-1]
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0].T
        elif arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        results: List[RawDetection] = []
        scale_x, scale_y = w / sz, h / sz
        for row in arr:
            if row.shape[0] < 5:
                continue
            obj_conf = float(row[4])
            if obj_conf < thr:
                continue
            class_id = int(np.argmax(row[5:]))
            score = obj_conf * float(row[5 + class_id])
            if score < thr:
                continue
            label = COCO_LABELS.get(class_id, str(class_id))
            if wanted_labels is not None and label not in wanted_labels:
                continue
            cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            x = (cx - bw / 2) / scale_x
            y = (cy - bh / 2) / scale_y
            rw = bw / scale_x
            rh = bh / scale_y
            if rw <= 0 or rh <= 0:
                continue
            results.append(RawDetection(label=label, confidence=score,
                                        x=x, y=y, width=rw, height=rh))
        return results, info
