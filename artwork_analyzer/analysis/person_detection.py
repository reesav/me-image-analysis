"""Person detection (full-body human figures).

Person detection and face detection are deliberately separate tasks. The
backend is pluggable; the default is the small SSD-MobileNet-v1 ONNX detector
(ONNX Runtime, Apache-2.0). Heavy detectors (YOLOv8) are optional plugins.
Detection is on photo-trained models and is weak on stylised/historical art —
manual review is expected. When no detector is configured or a model is
unavailable, an empty, explicitly 'unavailable' signal is returned.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from ..models.detectors import ORTSSDv1Backend, ORTYOLOv8Backend, RawDetection
from ..core.config import DetectorSettings
from ..core.schemas import PersonRecord, DetectionBox, ReviewStatus

logger = logging.getLogger("artwork_analyzer.analysis.person_detection")


def _person_backend(settings: DetectorSettings):
    kind = settings.person_detector
    if kind in ("disabled", "none", ""):
        return None
    if kind in ("onnx_ssd_mobilenet_v1", "ssd_mobilenet_v1"):
        return ORTSSDv1Backend(kind, conf_threshold=settings.person_confidence_threshold)
    if kind == "onnx_yolov8n":
        return ORTYOLOv8Backend(kind, conf_threshold=settings.person_confidence_threshold)
    raise ValueError(f"Unknown person detector: {kind}")


def detect_people(
    image_rgb: np.ndarray, settings: DetectorSettings
) -> Tuple[List[PersonRecord], str, str, bool]:
    """Detect people. Returns (records, backend_name, model_identifier, available)."""
    backend = _person_backend(settings)
    if backend is None:
        logger.info("Person detection disabled by settings.")
        return [], "disabled", "none", False

    raw, info = backend.detect(image_rgb, wanted_labels=["person"])
    if not info.available:
        logger.warning("Person detection model %s unavailable", settings.person_detector)

    records: List[PersonRecord] = []
    h, w = image_rgb.shape[:2]
    total_px = float(h * w)
    for i, det in enumerate(raw):
        if det.confidence < settings.person_confidence_threshold:
            continue
        box_px_area = det.width * det.height * total_px
        box = DetectionBox(
            x=det.x, y=det.y, width=det.width, height=det.height,
            confidence=det.confidence, label="person",
            review_status=ReviewStatus.AUTOMATED,
        )
        cx, cy = det.x + det.width / 2.0, det.y + det.height / 2.0
        records.append(
            PersonRecord(
                person_id=f"person_{i:03d}",
                box=box,
                box_area_percent=100.0 * box_px_area / total_px,
                centroid_x=cx,
                centroid_y=cy,
                review_status=ReviewStatus.AUTOMATED,
                detection_model=info.model_identifier,
            )
        )
    return records, info.name, info.model_identifier, info.available
