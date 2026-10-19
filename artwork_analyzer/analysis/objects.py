"""Optional model-based object/category detection.

Explicitly labelled as model-based and opt-in. Only runs when
``detector.object_detection_enabled`` is True. Returns generic
:class:`ObjectRecord` entries with stored confidence scores that can be
corrected by reviewers. Unsupported or uncertain categories are kept in the
record for human review rather than being folded into quantitative conclusions.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from ..models.detectors import ORTSSDv1Backend, ORTYOLOv8Backend
from ..core.config import DetectorSettings
from ..core.schemas import ObjectRecord, DetectionBox, ReviewStatus

logger = logging.getLogger("artwork_analyzer.analysis.objects")


def _object_backend(settings: DetectorSettings):
    kind = settings.object_detector
    if kind in ("onnx_ssd_mobilenet_v1", "ssd_mobilenet_v1"):
        return ORTSSDv1Backend(kind, conf_threshold=settings.object_confidence_threshold)
    if kind == "onnx_yolov8n":
        return ORTYOLOv8Backend(kind, conf_threshold=settings.object_confidence_threshold)
    raise ValueError(f"Unsupported object detector: {kind}")


def detect_objects(
    image_rgb: np.ndarray, settings: DetectorSettings
) -> Tuple[List[ObjectRecord], str, str, bool]:
    if not settings.object_detection_enabled:
        return [], "disabled", "none", False
    backend = _object_backend(settings)
    raw, meta = backend.detect(image_rgb, wanted_labels=None)
    records: List[ObjectRecord] = []
    for i, det in enumerate(raw):
        box = DetectionBox(
            x=det.x, y=det.y, width=det.width, height=det.height,
            confidence=det.confidence, label=det.label,
            review_status=ReviewStatus.AUTOMATED,
        )
        records.append(
            ObjectRecord(
                object_id=f"obj_{i:03d}",
                label=det.label,
                confidence=det.confidence,
                box=box,
                review_status=ReviewStatus.AUTOMATED,
            )
        )
    return records, meta.name, meta.model_identifier, meta.available
