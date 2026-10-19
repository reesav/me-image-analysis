"""Tests for deterministic colour measurements."""

from __future__ import annotations

import numpy as np
import pytest

from artwork_analyzer.analysis.color import (
    analyze_color, circular_mean_hue, colorfulness,
    dominant_colors, mean_median_rgb, channel_histograms,
)


def test_black_image_luminance(black_image):
    from artwork_analyzer.analysis.image_io import relative_luminance
    lum = relative_luminance(black_image)
    assert lum.mean() == pytest.approx(0.0, abs=1e-6)
    res = analyze_color(black_image)
    assert res["mean_luminance"] == pytest.approx(0.0, abs=1e-6)
    assert res["std_luminance"] == pytest.approx(0.0, abs=1e-6)


def test_white_image_luminance(white_image):
    from artwork_analyzer.analysis.image_io import relative_luminance
    lum = relative_luminance(white_image)
    assert lum.mean() == pytest.approx(1.0, abs=1e-6)
    res = analyze_color(white_image)
    assert res["mean_luminance"] == pytest.approx(1.0, abs=1e-6)
    assert res["max_luminance"] == pytest.approx(1.0, abs=1e-6)


def test_red_image_mean_rgb(red_image):
    mean_rgb, _ = mean_median_rgb(red_image)
    assert mean_rgb == pytest.approx([255.0, 0.0, 0.0], abs=0.5)


def test_hue_circular_mean_wraparound():
    # Hue values 350 and 10 degrees should wrap to ~0, not 180.
    hue = np.array([350.0, 10.0])
    m = circular_mean_hue(hue)
    assert m is not None
    assert 0.0 <= m <= 20.0  # close to 0, not 180


def test_colorfulness_zero_for_grayscale():
    gray = np.full((50, 50, 3), 128, dtype=np.uint8)
    assert colorfulness(gray) == pytest.approx(0.0, abs=1e-6)


def test_dominant_colors_deterministic(red_image):
    c1 = dominant_colors(red_image, n=1)
    c2 = dominant_colors(red_image, n=1)
    assert c1 == c2
    assert c1[0][0] > 200  # red channel dominant


def test_channel_histogram_sums_to_pixels(two_rectangles):
    hists = channel_histograms(two_rectangles)
    total = two_rectangles.shape[0] * two_rectangles.shape[1]
    for ch in "RGB":
        assert sum(hists[ch]) == total


def test_color_results_bounded(two_rectangles):
    res = analyze_color(two_rectangles)
    assert 0.0 <= res["mean_luminance"] <= 1.0
    assert res["colorfulness"] >= 0.0
    assert len(res["dominant_colors"]) == 5
