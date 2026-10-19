"""Shared pytest fixtures: synthetic images with known properties.

These fixtures underpin the validation suite so measurements can be checked
against mathematically known inputs (uniform colours, known rectangles,
overlapping masks, etc.).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture()
def black_image() -> np.ndarray:
    return np.zeros((128, 128, 3), dtype=np.uint8)


@pytest.fixture()
def white_image() -> np.ndarray:
    return np.full((128, 128, 3), 255, dtype=np.uint8)


@pytest.fixture()
def red_image() -> np.ndarray:
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[..., 0] = 255
    return img


@pytest.fixture()
def two_rectangles() -> np.ndarray:
    """Two clearly separated filled rectangles (person proxies)."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Left rectangle (red)
    img[40:160, 20:60, 0] = 200
    # Right rectangle (green)
    img[40:160, 140:180, 1] = 200
    return img


@pytest.fixture()
def overlapping_rectangles() -> np.ndarray:
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[40:160, 20:120, 0] = 200  # red-ish, left to middle
    img[60:180, 80:180, 1] = 200  # green-ish, middle to right (overlaps)
    return img


@pytest.fixture()
def face_like_image() -> np.ndarray:
    """A synthetic 'face-ish' cluster of pixels for detector smoke tests."""
    img = np.full((160, 160, 3), 200, dtype=np.uint8)
    # Two eyes (dark elliptical-ish blobs) and a mouth.
    cv2_circle = __import__("cv2").circle
    # left eye
    cv2_circle(img, (50, 50), 8, (0, 0, 0), -1)
    cv2_circle(img, (50, 50), 3, (200, 200, 200), -1)
    # right eye
    cv2_circle(img, (105, 50), 8, (0, 0, 0), -1)
    cv2_circle(img, (105, 50), 3, (200, 200, 200), -1)
    # mouth
    cv2_circle(img, (80, 105), 15, (0, 0, 0), -1)
    cv2_circle(img, (80, 105), 9, (200, 200, 200), -1)
    # nose
    cv2_circle(img, (80, 80), 6, (80, 80, 80), -1)
    return img


@pytest.fixture()
def tmp_image_file(tmp_path: Path, two_rectangles: np.ndarray) -> Path:
    import cv2
    p = tmp_path / "stimulus.png"
    cv2.imwrite(str(p), cv2.cvtColor(two_rectangles, cv2.COLOR_RGB2BGR))
    return p


@pytest.fixture()
def gradient_image() -> np.ndarray:
    """Horizontal 0-255 luminance gradient."""
    row = np.linspace(0, 255, 200, dtype=np.uint8)
    return np.tile(row, (200, 1))[:, :, None].repeat(3, axis=2)


@pytest.fixture()
def data_dir() -> Path:
    from artwork_analyzer.models.registry import PACKAGE_DATA_DIR
    return PACKAGE_DATA_DIR
