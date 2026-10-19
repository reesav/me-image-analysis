"""Tests for the full image analysis orchestrator and result schema."""

from __future__ import annotations

import numpy as np
import pytest

from artwork_analyzer.analysis.analyzer import analyze_image
from artwork_analyzer.core.config import AppSettings, DetectorSettings


def test_analyze_image_completes(tmp_path, two_rectangles):
    # Make an image with EXIF orientation 6.
    from PIL import Image
    import cv2
    p = tmp_path / "stim.tiff"
    bgr = cv2.cvtColor(two_rectangles, cv2.COLOR_RGB2BGR)
    img = Image.fromarray(bgr)
    exif = img.getexif(); exif[274] = 6
    img.save(str(p), "TIFF", exif=exif.tobytes())

    settings = AppSettings(detector=DetectorSettings(person_detector="disabled"))
    res = analyze_image(str(p), settings, stimulus_id="STIM_001")

    assert res.completed is True
    assert res.media_type.value == "image"
    assert res.image_metadata["width"] == 200
    assert res.image_metadata["height"] == 200
    assert res.image_metadata["exif_orientation"] == 6
    assert res.color is not None and res.color.mean_luminance >= 0.0
    assert res.brightness_contrast is not None
    assert res.composition is not None
    assert res.reproducibility.app_version == "1.0.0"
    assert res.reproducibility.person_detector_model == "none"
    assert "color" in res.reproducibility.analysis_module_versions


def test_analyze_image_failure_recorded(tmp_path):
    p = tmp_path / "bad.png"
    p.write_bytes(b"corrupt")
    settings = AppSettings()
    res = analyze_image(str(p), settings, stimulus_id="BAD")
    assert res.completed is False
    assert any("image_load_failed" in e for e in res.errors)


def test_analyze_image_records_dependency_versions(tmp_path, red_image):
    import cv2
    p = tmp_path / "r.png"
    cv2.imwrite(str(p), cv2.cvtColor(red_image, cv2.COLOR_RGB2BGR))
    settings = AppSettings(detector=DetectorSettings(person_detector="disabled"))
    res = analyze_image(str(p), settings)
    assert "numpy" in res.reproducibility.dependency_versions
    assert "opencv" in res.reproducibility.dependency_versions
