"""Deterministic colour measurements.

All functions are pure and operate on RGB uint8 arrays so they are fully
reproducible and unit-testable. Colour-space conversions and aggregations are
defined explicitly in :mod:`artwork_analyzer.analysis.image_io`.

Metric definitions
------------------
- RGB means/medians: arithmetic means of each 0-255 channel.
- HSV: converted via standard max-min formula; H reported in degrees (0-360),
  S and V in 0-1. Hue is aggregated with a circular mean.
- Luminance: sRGB relative luminance (ITU-R BT.709), range 0-1.
- Colourfulness: Hasler & Susstrunk "Peron's" colourfulness (2003), computed
  in RGB space via the ``r-g``/``b-y`` opponent transform; a well-defined,
  bounded metric. Not a validated psychological "colorfulness" construct.
- Dominant colours: k-means (k fixed, deterministic seed) over RGB pixels.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .image_io import rgb_to_hsv, circular_mean_hue, relative_luminance

DEFAULT_HIST_BINS = 256


def channel_histograms(image_rgb: np.ndarray, bins: int = DEFAULT_HIST_BINS) -> Dict[str, List[int]]:
    """Return 256-bin histograms for each channel as plain lists."""
    hist: Dict[str, List[int]] = {}
    for i, name in enumerate("RGB"):
        h, _ = np.histogram(image_rgb[..., i].astype(np.int32), bins=bins, range=(0, 255))
        hist[name] = h.tolist()
    return hist


def mean_median_rgb(image_rgb: np.ndarray) -> Tuple[List[float], List[float]]:
    flat = image_rgb.reshape(-1, 3).astype(np.float64)
    return flat.mean(axis=0).tolist(), np.median(flat, axis=0).tolist()


def mean_median_hsv(image_rgb: np.ndarray) -> Tuple[List[float], List[float]]:
    hsv = rgb_to_hsv(image_rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # Hue: circular mean; Saturation/Value: arithmetic mean & median.
    circ = circular_mean_hue(h) or 0.0
    mean_hsv = [circ, float(np.mean(s)), float(np.mean(v))]
    med_hsv = [circ, float(np.median(s)), float(np.median(v))]
    return mean_hsv, med_hsv


def luminance_stats(lum: np.ndarray) -> Dict[str, float]:
    flat = lum.ravel().astype(np.float64)
    return {
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "p10": float(np.percentile(flat, 10)),
        "p25": float(np.percentile(flat, 25)),
        "p75": float(np.percentile(flat, 75)),
        "p90": float(np.percentile(flat, 90)),
    }


def colorfulness(image_rgb: np.ndarray) -> float:
    """Hasler-Susstrunk perceived colourfulness.

    Computes the opponent colour axes (R-G, B-Y), then the metric
    ``M = mean(sqrt(sigma_a^2 + sigma_b^2)) + 0.3 * sqrt(mean_a^2 + mean_b^2)``.
    """
    rgb = image_rgb.astype(np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    a = r - g
    b_y = b - (r + g) / 2.0
    sigma_a = np.std(a)
    sigma_b = np.std(b_y)
    mean_a = np.mean(a)
    mean_b = np.mean(b_y)
    return float(np.sqrt(sigma_a**2 + sigma_b**2) + 0.3 * np.sqrt(mean_a**2 + mean_b**2))


def color_proportions(
    image_rgb: np.ndarray, ranges: Optional[List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = None
) -> Dict[str, float]:
    """Proportion of pixels falling within each configurable RGB cube range.

    ``ranges`` is a list of (low, high) RGB triples (0-255). Default set is a
    small named palette useful for art research.
    """
    if ranges is None:
        ranges = _DEFAULT_COLOR_RANGES

    total = image_rgb.shape[0] * image_rgb.shape[1]
    if total == 0:
        return {name: 0.0 for name, _ in _DEFAULT_COLOR_RANGES}

    out: Dict[str, float] = {}
    for name, (low, high) in ranges:
        low_arr = np.array(low, dtype=np.int16)
        high_arr = np.array(high, dtype=np.int16)
        img_i16 = image_rgb.astype(np.int16)
        mask = np.all((img_i16 >= low_arr) & (img_i16 <= high_arr), axis=2)
        out[name] = float(np.count_nonzero(mask) / total)
    return out


_DEFAULT_COLOR_RANGES: List[Tuple[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = [
    ("red", ((150, 0, 0), (255, 80, 80))),
    ("green", ((0, 150, 0), (80, 255, 80))),
    ("blue", ((0, 0, 150), (80, 80, 255))),
    ("black", ((0, 0, 0), (30, 30, 30))),
    ("white", ((225, 225, 225), (255, 255, 255))),
    ("yellow", ((200, 200, 0), (255, 255, 120))),
    ("gray", ((80, 80, 80), (175, 175, 175))),
]


def _kmeans_numpy(pixels: np.ndarray, n: int, seed: int = 42, iters: int = 50) -> np.ndarray:
    """Deterministic k-means on float64 pixel rows using a seeded RNG."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pixels), size=n, replace=len(pixels) >= n)
    centroids = pixels[idx].copy()
    if len(pixels) < n:
        # Pad with duplicates if we need more centroids than pixels.
        centroids = np.resize(centroids, (n, pixels.shape[1]))
    labels = np.zeros(len(pixels), dtype=np.intp)
    for _ in range(iters):
        prev = centroids.copy()
        dists = np.linalg.norm(pixels[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        for k in range(n):
            members = pixels[labels == k]
            if len(members) > 0:
                centroids[k] = members.mean(axis=0)
        if np.allclose(centroids, prev):
            break
    return centroids


def dominant_colors(image_rgb: np.ndarray, n: int = 5, seed: int = 42) -> List[List[float]]:
    """Return top-N dominant RGB colours via k-means (deterministic seed)."""
    pixels = image_rgb.reshape(-1, 3).astype(np.float64)
    n = max(1, min(n, len(pixels)))
    if len(pixels) <= n:
        uniq = np.unique(pixels.astype(np.uint8), axis=0)
        return uniq[:n].tolist()
    centroids = _kmeans_numpy(pixels, n, seed=seed)
    # Order by cluster size (largest first) for a meaningful "dominant" ranking.
    dists = np.linalg.norm(pixels[:, None, :] - centroids[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)
    sizes = np.bincount(labels, minlength=n)
    order = np.argsort(-sizes)
    return [centroids[k].round(1).tolist() for k in order]


def analyze_color(image_rgb: np.ndarray, n_dominant: int = 5) -> dict:
    """Run all deterministic colour measurements and return a plain dict."""
    mean_rgb, med_rgb = mean_median_rgb(image_rgb)
    mean_hsv, med_hsv = mean_median_hsv(image_rgb)
    lum = relative_luminance(image_rgb)
    lum_stats = luminance_stats(lum)
    hsv = rgb_to_hsv(image_rgb)
    circ_hue = circular_mean_hue(hsv[..., 0])

    return {
        "mean_rgb": mean_rgb,
        "median_rgb": med_rgb,
        "mean_hsv": mean_hsv,
        "median_hsv": med_hsv,
        "mean_luminance": lum_stats["mean"],
        "median_luminance": lum_stats["median"],
        "std_luminance": lum_stats["std"],
        "min_luminance": lum_stats["min"],
        "max_luminance": lum_stats["max"],
        "p25_luminance": lum_stats["p25"],
        "p75_luminance": lum_stats["p75"],
        "p10_luminance": lum_stats["p10"],
        "p90_luminance": lum_stats["p90"],
        "hue_circular_mean": circ_hue,
        "colorfulness": colorfulness(image_rgb),
        "channel_histograms": channel_histograms(image_rgb),
        "color_proportions": color_proportions(image_rgb),
        "dominant_colors": dominant_colors(image_rgb, n=n_dominant),
    }
