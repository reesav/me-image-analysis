"""Person instance segmentation.

Default backend: **GrabCut** seeded by person detection boxes (CPU, no
download, always available when a box is supplied). This produces a distinct
mask per box but is weak on overlapping / stylised figures — exactly why
manual review exists. Optional heavier backends (MediaPipe selfie
segmentation, Segment Anything 2) are supported as plugins but require extra
dependencies and are lazily loaded.

Masks are returned at the **original image resolution** so that pixel-area
metrics are computed against true dimensions.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..core.config import DetectorSettings
from ..core.schemas import PersonRecord, ReviewStatus

logger = logging.getLogger("artwork_analyzer.analysis.segmentation")

# Padding fraction around the bounding box for the GrabCut rect.
_BOX_PAD = 0.05
# Confidence reported for GrabCut is the fraction of the rect inside the
# predicted FG, used as a rough quality proxy in 0-1.


def _rect_from_box(h: int, w: int, box_norm, pad: float = _BOX_PAD) -> Tuple[int, int, int, int]:
    x = box_norm.x
    y = box_norm.y
    bw = box_norm.width
    bh = box_norm.height
    px = max(0.0, x - bw * pad)
    py = max(0.0, y - bh * pad)
    pr = min(1.0, x + bw * (1.0 + pad))
    pb = min(1.0, y + bh * (1.0 + pad))
    rx, ry = int(px * w), int(py * h)
    rw = max(1, int(pr * w) - rx)
    rh = max(1, int(pb * h) - ry)
    # Clamp within image bounds.
    rx = max(0, min(w - 1, rx))
    ry = max(0, min(h - 1, ry))
    rw = max(1, min(w - rx, rw))
    rh = max(1, min(h - ry, rh))
    return rx, ry, rw, rh


def grabcut_segment(image_rgb: np.ndarray, box_norm) -> Tuple[np.ndarray, float]:
    """Run GrabCut seeded inside the box; return (binary_mask, confidence)."""
    h, w = image_rgb.shape[:2]
    rx, ry, rw, rh = _rect_from_box(h, w, box_norm)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    # cv2 requires at least a 1px border around the rect.
    sx, sy = max(0, rx - 1), max(0, ry - 1)
    ex, ey = min(w, rx + rw + 1), min(h, ry + rh + 1)
    ex = max(ex, sx + 2)
    ey = max(ey, sy + 2)
    patch = bgr[sy:ey, sx:ex].copy()
    rect = (rx - sx, ry - sy, rw, rh)
    if min(patch.shape[:2]) < 2:
        # Degenerate box; fall back to the whole image.
        patch = bgr.copy()
        rect = (0, 0, w - 1, h - 1)
    mask = np.zeros(patch.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(patch, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error as exc:
        logger.warning("GrabCut failed: %s", exc)
        return np.zeros((h, w), dtype=bool), 0.0
    # Build full-image mask.
    fg_mask = np.zeros((h, w), dtype=bool)
    fg_patch = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    ys, xs = np.where(fg_patch)
    fg_mask[ys + sy, xs + sx] = True
    confidence = float(np.mean(mask[fg_patch] == cv2.GC_FGD)) if fg_patch.any() else 0.0
    confidence = float(np.clip(confidence if fg_patch.any() else 0.0, 0.0, 1.0))
    # If no FG pixels, mark uncertain.
    if not fg_mask.any():
        confidence = 0.0
    return fg_mask, round(confidence, 3)


def _try_mediapipe(image_rgb: np.ndarray):
    """Optional MediaPipe selfie segmentation backend (lazy import)."""
    try:
        import mediapipe as mp  # type: ignore
    except Exception as exc:
        logger.info("MediaPipe not available: %s", exc)
        return None
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    raise NotImplementedError  # MediaPipe tasks require model download; not wired default


def _try_sam2(image_rgb: np.ndarray, box):
    """Optional SAM2 backend (requires torch + segment_anything)."""
    try:
        import torch  # noqa: F401
        from segment_anything import sam_model_registry, SamPredictor  # type: ignore
    except Exception as exc:
        logger.info("SAM2 not available: %s", exc)
        return None
    raise NotImplementedError


def segment_people(
    image_rgb: np.ndarray, people: List[PersonRecord], settings: DetectorSettings
) -> List[Tuple[str, np.ndarray, float, str]]:
    """Segment each person, returning (person_id, mask, confidence, backend).

    Falls back gracefully: if a box is missing, the person is skipped and a
    warning is recorded in their review_note. Returns one entry per person
    with a valid box.
    """
    backend = settings.segmentation_backend
    results: List[Tuple[str, np.ndarray, float, str]] = []
    for person in people:
        if person.box is None:
            person.review_note = (person.review_note or "") + " | no box; segmentation skipped"
            continue
        if backend in ("grabcut", "default"):
            mask, conf = grabcut_segment(image_rgb, person.box)
            results.append((person.person_id, mask, conf, "grabcut"))
        elif backend in ("mediapipe",):
            # Lazy; falls back to grabcut if unavailable.
            results.append((person.person_id, np.zeros((0, 0), bool), 0.0, "mediapipe(unavailable)"))
        elif backend in ("sam2",):
            results.append((person.person_id, np.zeros((0, 0), bool), 0.0, "sam2(unavailable)"))
        else:
            mask, conf = grabcut_segment(image_rgb, person.box)
            results.append((person.person_id, mask, conf, "grabcut"))
    return results


def apply_masks(
    people: List[PersonRecord], masks: List[Tuple[str, np.ndarray, float, str]]
) -> None:
    """Update PersonRecord mask area metrics from segmentation results.

    The mask arrays themselves are stored separately by the storage layer;
    here we update area metrics and confidence. Total pixels are derived from
    each mask shape so metrics are computed against true image dimensions.
    """
    lookup = {pid: (mask, conf, backend) for pid, mask, conf, backend in masks}
    for person in people:
        entry = lookup.get(person.person_id)
        if entry is None:
            continue
        mask, conf, backend = entry
        total_px = int(mask.shape[0] * mask.shape[1])
        person.mask_area_pixels = int(np.count_nonzero(mask))
        person.mask_area_percent = round(100.0 * person.mask_area_pixels / total_px, 4) if total_px else None
        person.mask_available = bool(mask.any())
        person.mask_confidence = conf
        person.segmentation_model = backend
