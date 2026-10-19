"""Brightness and contrast measurements.

Each contrast definition is reported under an explicit name so that they are
never conflated. Luminance is sRGB relative luminance in 0-1 range.

Metric definitions
------------------
- ``rms_contrast``: root-mean-square contrast = standard deviation of the
  normalised luminance (Reinhard, 2013). Reported as a fraction (0-1) and is
  dimensionless.
- ``michelson_contrast``: (L_max - L_min) / (L_max + L_min). Only reported
  when ``michelson_applicable`` is True, which requires a bimodal luminance
  distribution (a clear bright region and a clear dark region). This avoids
  misapplying the metric to near-uniform images.
- ``local_contrast``: mean and std of per-tile RMS contrast (non-overlapping
  tiles). Reflects spatial variability of contrast, not a single global value.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .image_io import relative_luminance


def _lum_array(image_rgb: np.ndarray) -> np.ndarray:
    return relative_luminance(image_rgb)


def _percentiles(flat: np.ndarray) -> Dict[str, float]:
    return {
        "p10": float(np.percentile(flat, 10)),
        "p25": float(np.percentile(flat, 25)),
        "p75": float(np.percentile(flat, 75)),
        "p90": float(np.percentile(flat, 90)),
    }


def _bimodal_heuristic(lum: np.ndarray) -> bool:
    """Heuristic: True if luminance has distinct dark and bright regions.

    Uses Otsu-style separation: the image is considered to have a usable dark
    region and bright region if the 5th percentile is below a quarter of the
    95th percentile AND the 95th percentile exceeds 0.15. This prevents
    Michelson being reported for near-uniform images where the formula is
    meaningless.
    """
    p05 = float(np.percentile(lum, 5))
    p95 = float(np.percentile(lum, 95))
    if p95 < 0.15:
        return False
    if p05 >= 0.25 * p95:
        return False
    return True


def analyze_brightness_contrast(image_rgb: np.ndarray, tile_size: int = 64) -> dict:
    lum = _lum_array(image_rgb).ravel().astype(np.float64)
    mean_l = float(np.mean(lum))
    std_l = float(np.std(lum))

    rms_contrast = std_l  # dimensionless std of normalised luminance
    lmin = float(np.min(lum))
    lmax = float(np.max(lum))
    michelson_applicable = _bimodal_heuristic(lum)
    michelson = (
        float((lmax - lmin) / (lmax + lmin)) if (michelson_applicable and (lmax + lmin) > 0) else None
    )

    # Local contrast: per-tile RMS contrast.
    h, w = image_rgb.shape[:2]
    tile = max(1, min(tile_size, h, w))
    local_values = []
    lum2 = relative_luminance(image_rgb)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            block = lum2[y : y + tile, x : x + tile]
            if block.size < 4:
                continue
            local_values.append(float(np.std(block)))
    if local_values:
        lv = np.array(local_values, dtype=np.float64)
        local_mean = float(np.mean(lv))
        local_std = float(np.std(lv))
    else:
        local_mean = None
        local_std = None

    stats = _percentiles(lum)
    return {
        "mean_luminance": mean_l,
        "median_luminance": float(np.median(lum)),
        "std_luminance": std_l,
        "min_luminance": lmin,
        "max_luminance": lmax,
        "p10_luminance": stats["p10"],
        "p25_luminance": stats["p25"],
        "p75_luminance": stats["p75"],
        "p90_luminance": stats["p90"],
        "rms_contrast": rms_contrast,
        "michelson_contrast": michelson,
        "michelson_applicable": michelson_applicable,
        "local_contrast_mean": local_mean,
        "local_contrast_std": local_std,
        "tile_size": tile,
    }
