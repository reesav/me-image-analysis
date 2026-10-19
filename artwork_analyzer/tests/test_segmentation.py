"""Tests for person segmentation (GrabCut) on synthetic figures."""

from __future__ import annotations

import numpy as np
import pytest

from artwork_analyzer.analysis.segmentation import (
    grabcut_segment, segment_people, apply_masks,
)
from artwork_analyzer.core.schemas import PersonRecord, DetectionBox, ReviewStatus
from artwork_analyzer.core.config import DetectorSettings


def _person_from_box(x, y, w, h, pid="person_000"):
    return PersonRecord(
        person_id=pid,
        box=DetectionBox(x=x, y=y, width=w, height=h, confidence=0.9,
                         label="person", review_status=ReviewStatus.AUTOMATED),
        box_area_percent=0.0,
    )


def test_grabcut_on_solid_rectangle():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[40:160, 20:180, 0] = 200  # a red-ish rectangle
    box = DetectionBox(x=0.1, y=0.2, width=0.8, height=0.6, confidence=0.9, label="person")
    mask, conf = grabcut_segment(img, box)
    assert mask.shape == (200, 200)
    assert mask.sum() > 0  # some foreground recovered
    total = img.shape[0] * img.shape[1]
    assert 0.0 < (mask.sum() / total) < 1.0


def test_segment_people_and_areas(two_rectangles):
    settings = DetectorSettings(segmentation_backend="grabcut")
    p1 = _person_from_box(0.10, 0.20, 0.20, 0.70, "person_000")
    p2 = _person_from_box(0.70, 0.20, 0.20, 0.70, "person_001")
    masks = segment_people(two_rectangles, [p1, p2], settings)
    assert len(masks) == 2
    apply_masks([p1, p2], masks)
    for p in (p1, p2):
        assert p.mask_available
        assert p.mask_area_percent is not None
        assert 0.0 < p.mask_area_percent < 100.0
    # Two separate figures should not be merged.
    assert p1.mask_area_pixels > 0 and p2.mask_area_pixels > 0


def test_segmentation_skips_missing_box():
    settings = DetectorSettings()
    p = PersonRecord(person_id="p0", box=None)  # type: ignore[arg-type]
    masks = segment_people(np.zeros((50, 50, 3), np.uint8), [p], settings)
    assert masks == []
