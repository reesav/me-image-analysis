"""Tests for brightness / contrast measurements."""

from __future__ import annotations

import numpy as np
import pytest

from artwork_analyzer.analysis.brightness_contrast import analyze_brightness_contrast
from artwork_analyzer.analysis.image_io import relative_luminance


def test_uniform_image_zero_contrast(black_image, white_image):
    for img in (black_image, white_image):
        res = analyze_brightness_contrast(img)
        assert res["rms_contrast"] == pytest.approx(0.0, abs=1e-6)
        assert res["michelson_contrast"] is None
        assert res["michelson_applicable"] is False


def test_gradient_has_consistent_contrast(gradient_image):
    res = analyze_brightness_contrast(gradient_image)
    # A horizontal gradient: rms contrast equals std of luminance.
    lum = relative_luminance(gradient_image)
    expected = float(lum.std())
    assert res["rms_contrast"] == pytest.approx(expected, rel=1e-4)
    assert res["mean_luminance"] == pytest.approx(float(lum.mean()), rel=1e-4)
    assert res["min_luminance"] == pytest.approx(0.0, abs=1e-6)
    assert res["max_luminance"] == pytest.approx(1.0, abs=1e-6)


def test_michelson_for_bimodal(two_rectangles):
    res = analyze_brightness_contrast(two_rectangles)
    # Two bright rectangles on black -> bimodal -> michelson reported.
    assert res["michelson_applicable"] is True
    expected = (res["max_luminance"] - res["min_luminance"]) / (res["max_luminance"] + res["min_luminance"])
    assert res["michelson_contrast"] == pytest.approx(expected, rel=1e-4)


def test_local_contrast_present(gradient_image):
    res = analyze_brightness_contrast(gradient_image)
    # Gradient should have nonzero local contrast variability.
    assert res["local_contrast_mean"] is not None
    assert res["local_contrast_mean"] >= 0.0
