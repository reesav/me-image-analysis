"""Composition and spatial-distribution metrics.

Foreground masks are required for spatial metrics. When instance segmentation
of people is unavailable, a deterministic coarse foreground estimate is
produced with a blur-difference saliency proxy (Gaussian-blur background
subtraction) + Otsu threshold. These are explicitly labelled as coarse proxies,
not psychological constructs.
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

from .image_io import relative_luminance

# Saliency blur kernel is derived from image size (a fraction of the short edge).
_BLUR_FRACTION = 0.05
_MIN_BLUR_KSIZE = 7


def foreground_mask(coarse: np.ndarray) -> np.ndarray:
    """Produce a binary foreground mask from a saliency/probability map (0-1)."""
    sal = np.clip(coarse, 0.0, 1.0)
    # Morphological smoothing to merge fragmented detections.
    uint = (sal * 255).astype(np.uint8)
    uint = cv2.GaussianBlur(uint, (5, 5), 0)
    if uint.max() == 0:
        binary = np.zeros_like(uint, dtype=np.uint8)
    else:
        binary = cv2.inRange(uint, int(sal.std() * 255), 255)
    # Fallback Otsu if the adaptive threshold produced nothing.
    if float(binary.max()) == 0:
        _, binary = cv2.threshold(uint, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return (binary > 0).astype(bool)


def saliency_coarse(image_rgb: np.ndarray) -> np.ndarray:
    """Blur-difference coarse saliency, normalised 0-1 (deterministic)."""
    small = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = small.shape
    ksize = max(_MIN_BLUR_KSIZE, int(_BLUR_FRACTION * min(h, w)) | 1)
    blurred = cv2.GaussianBlur(small, (ksize, ksize), 0)
    sal = np.abs(small - blurred)
    mx = float(sal.max())
    if mx > 0:
        sal = sal / mx
    return sal.astype(np.float32)


def _distribution(frac_of_edge: int, mask: np.ndarray) -> np.ndarray:
    """Fraction of foreground pixels along strips parallel to the axis."""
    h, w = mask.shape
    if mask.sum() == 0:
        return np.zeros(frac_of_edge, dtype=float)
    if frac_of_edge == h and frac_of_edge > 1:
        # vertical distribution means distribution across rows
        per_row = mask.sum(axis=1) / max(mask.sum(), 1)
        return np.mean(per_row.reshape(frac_of_edge, -1), axis=1)
    return np.array([])


def analyze_composition(image_rgb: np.ndarray, n_strips: int = 6, n_grid: int = 6) -> dict:
    sal = saliency_coarse(image_rgb)
    mask = foreground_mask(sal)
    h, w = mask.shape
    total_pixels = mask.size
    fg_pixels = int(mask.sum())
    fg_ratio = fg_pixels / total_pixels if total_pixels else 0.0

    # Horizontal distribution: fraction of foreground per vertical band
    # (left-to-right). Each value = share of total foreground in that column band.
    vert_bands = np.array_split(np.arange(w), n_strips)
    horiz_dist = []
    for band in vert_bands:
        cnt = int(mask[:, band].sum())
        horiz_dist.append(cnt / fg_pixels if fg_pixels else 0.0)

    horiz_bands = np.array_split(np.arange(h), n_strips)
    vert_dist = []
    for band in horiz_bands:
        cnt = int(mask[band, :].sum())
        vert_dist.append(cnt / fg_pixels if fg_pixels else 0.0)

    # Grid occupancy (fraction of foreground per cell).
    grid = np.zeros((n_grid, n_grid), dtype=float)
    row_edges = np.linspace(0, h, n_grid + 1).astype(int)
    col_edges = np.linspace(0, w, n_grid + 1).astype(int)
    for i in range(n_grid):
        for j in range(n_grid):
            cell = mask[row_edges[i] : row_edges[i + 1], col_edges[j] : col_edges[j + 1]]
            grid[i, j] = float(cell.sum()) / fg_pixels if fg_pixels else 0.0

    # Centre of mass of foreground (normalised 0-1, origin top-left).
    if fg_pixels:
        ys, xs = np.where(mask)
        com_x = float(xs.mean()) / w
        com_y = float(ys.mean()) / h
    else:
        com_x, com_y = 0.5, 0.5

    # Rule-of-thirds alignment: proximity of COM to the two vertical and two
    # horizontal rule-of-thirds lines (distance normalised to 0-1, 0 = on line).
    third = 1.0 / 3.0
    rots = [
        1.0 - min(abs(com_x - third), abs(com_x - 2.0 * third)) / third,
        1.0 - min(abs(com_y - third), abs(com_y - 2.0 * third)) / third,
    ]

    return {
        "horizontal_foreground_distribution": [round(v, 4) for v in horiz_dist],
        "vertical_foreground_distribution": [round(v, 4) for v in vert_dist],
        "grid_occupancy": [[round(float(v), 4) for v in row] for row in grid],
        "center_of_mass_x": com_x,
        "center_of_mass_y": com_y,
        "foreground_background_ratio": fg_ratio,
        "rule_of_thirds_alignment": [round(v, 4) for v in rots],
        "foreground_mask_available": True,
        "foreground_pixels": fg_pixels,
    }
