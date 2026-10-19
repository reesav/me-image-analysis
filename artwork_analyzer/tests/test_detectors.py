"""Tests for detector backends and model availability/licensing."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from artwork_analyzer.models.registry import resolve_model, model_report
from artwork_analyzer.models.detectors import YuNetBackend, ORTSSDv1Backend
from artwork_analyzer.analysis.face_detection import detect_faces
from artwork_analyzer.analysis.person_detection import detect_people
from artwork_analyzer.core.config import DetectorSettings


def test_bundled_yunet_model_present():
    path = resolve_model("face_yunet")
    assert path is not None, "YuNet model not bundled"
    assert path.exists()


def test_model_report_lists_licenses():
    report = model_report()
    assert "face_yunet" in report
    assert report["face_yunet"]["license"] == "Apache-2.0 (OpenCV)"
    assert report["onnx_ssd_mobilenet_v1"]["license"] == "Apache-2.0 (ONNX Model Zoo)"
    assert report["onnx_yolov8n"]["license"] == "AGPL-3.0-or-later (Ultralytics)"


def test_face_detector_runs(face_like_image):
    settings = DetectorSettings()
    records, backend, model, available = detect_faces(face_like_image, settings)
    assert isinstance(records, list)
    assert backend == "yunet"
    assert available is True
    for rec in records:
        assert 0.0 <= rec.box.x <= 1.0
        assert 0.0 <= rec.box.width <= 1.0
        assert 0.0 <= rec.box.confidence <= 1.0


def test_face_detector_real_photo():
    from pathlib import Path
    p = Path("/tmp/bench/American_Gothic.jpg")
    if not p.exists():
        pytest.skip("benchmark image unavailable")
    img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    records, backend, model, available = detect_faces(img, DetectorSettings())
    assert available
    assert len(records) >= 1  # American Gothic has two faces


def test_person_detector_disabled(two_rectangles):
    settings = DetectorSettings(person_detector="disabled")
    records, backend, model, available = detect_people(two_rectangles, settings)
    assert available is False
    assert records == []


def test_ssd_backend_optional(two_rectangles):
    path = resolve_model("onnx_ssd_mobilenet_v1")
    if path is None:
        pytest.skip("SSD model not cached")
    settings = DetectorSettings(person_detector="onnx_ssd_mobilenet_v1")
    records, backend, model, available = detect_people(two_rectangles, settings)
    assert available is True
    assert all(r.box.box_area_percent >= 0.0 for r in records)
