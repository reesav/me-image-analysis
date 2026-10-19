"""Tests for composition / spatial-distribution metrics."""

from __future__ import annotations

import numpy as np
import pytest

from artwork_analyzer.analysis.composition import analyze_composition


def test_composition_returns_valid_ranges(two_rectangles):
    res = analyze_composition(two_rectangles)
    assert sum(res["horizontal_foreground_distribution"]) == pytest.approx(1.0, abs=1e-3)
    assert sum(res["vertical_foreground_distribution"]) == pytest.approx(1.0, abs=1e-3)
    assert 0.0 <= res["center_of_mass_x"] <= 1.0
    assert 0.0 <= res["center_of_mass_y"] <= 1.0
    assert res["foreground_background_ratio"] >= 0.0


def test_composition_empty_image_does_not_crash(black_image):
    res = analyze_composition(black_image)
    assert res["foreground_background_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert res["center_of_mass_x"] == 0.5


def test_composition_grid_sums(two_rectangles):
    res = analyze_composition(two_rectangles, n_grid=6)
    grid = np.array(res["grid_occupancy"])
    assert grid.shape == (6, 6)
    assert abs(grid.sum() - 1.0) < 1e-3 or grid.sum() == pytest.approx(1.0, abs=1e-3)
