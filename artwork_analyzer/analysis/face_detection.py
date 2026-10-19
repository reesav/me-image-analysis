"""Face detection (distinct from person detection).

Default backend: OpenCV ``FaceDetectorYN`` with the bundled YuNet ONNX model
(Apache-2.0). Model-based predictions are stored separately from manual
corrections. Faces near the image border are flagged ``is_cropped``.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from ..models.detectors import YuNetBackend, RawDetection, DetectionBackend
from ..core.config import DetectorSettings
from ..core.schemas import FaceRecord, DetectionBox, ReviewStatus

logger = logging.getLogger("artwork_analyzer.analysis.face_detection")

_CROP_MARGIN = 0.02


def _is_cropped(det: RawDetection) -> bool:
    margin = _CROP_MARGIN
    touches = (
        det.x < margin or det.y < margin or
        (det.x + det.width) > 1.0 - margin or
        (det.y + det.height) > 1.0 - margin
    )
    too_small = det.width < 0.03 or det.height < 0.03
    return bool(touches or too_small)


def _face_backend(settings: DetectorSettings):
    kind = settings.face_detector
    if kind in ("face_yunet", "yunet", "face"):
        return YuNetBackend(
            conf_threshold=settings.face_confidence_threshold,
            score_threshold=settings.face_score_threshold,
            nms_threshold=settings.face_nms_threshold,
        )
    raise ValueError(f"Unknown face detector: {kind}")


def detect_faces(
    image_rgb: np.ndarray, settings: DetectorSettings
) -> Tuple[List[FaceRecord], str, str, bool]:
    """Detect faces; returns (records, backend_name, model_identifier, available)."""
    backend = _face_backend(settings)
    raw, info = backend.detect(image_rgb)
    h, w = image_rgb.shape[:2]
    total_px = float(h * w)
    records: List[FaceRecord] = []
    for i, det in enumerate(raw):
        box_px_area = det.width * det.height * total_px
        box = DetectionBox(
            x=det.x, y=det.y, width=det.width, height=det.height,
            confidence=det.confidence, label="face",
            review_status=ReviewStatus.AUTOMATED,
        )
        records.append(
            FaceRecord(
                face_id=f"face_{i:03d}",
                box=box,
                box_area_percent=100.0 * box_px_area / total_px,
                is_cropped=_is_cropped(det),
                review_status=ReviewStatus.AUTOMATED,
                detection_model=info.model_identifier,
            )
        )
    return records, info.name, info.model_identifier, info.available
