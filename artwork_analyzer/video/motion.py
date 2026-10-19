"""Motion metrics via dense optical flow (Farneback), CPU-only.

Produces mean/median/std/p90 motion magnitude, the proportion of sampled
frames exceeding a movement threshold, and per-frame magnitudes.

IMPORTANT: dense optical flow conflates camera motion with subject motion.
The documentation states that these metrics do NOT isolate movement of
depicted subjects. ``motion_block_size`` downscales frames before flow for
speed. Magnitudes are normalised by the frame diagonal so the metric is in a
rough "fraction of frame size" unit.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger("artwork_analyzer.video.motion")


def _downscale(frame_gray: np.ndarray, block: int) -> np.ndarray:
    h, w = frame_gray.shape
    scale = max(1, block)
    return cv2.resize(frame_gray, (max(2, w // scale), max(2, h // scale)))


def sample_motion(
    path: str,
    frame_interval_seconds: float = 1.0,
    block_size: int = 16,
    min_magnitude: float = 2.0,
    max_frames: int = 0,
) -> List[float]:
    """Return per-frame motion magnitude at the given sampling interval."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    step = max(1, int(round(frame_interval_seconds * fps))) if fps > 0 else 1
    magnitudes: List[float] = []
    prev_small: Optional[np.ndarray] = None
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and idx >= max_frames:
            break
        if idx % step != 0:
            idx += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = _downscale(gray, block_size)
        diag = float(np.sqrt(small.shape[0] ** 2 + small.shape[1] ** 2))
        if prev_small is not None:
            # Ensure same shape.
            if prev_small.shape != small.shape:
                prev_small = cv2.resize(prev_small, (small.shape[1], small.shape[0]))
            flow = cv2.calcOpticalFlowFarneback(
                prev_small, small, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            magnitudes.append(float(np.mean(mag)) / diag if diag > 0 else 0.0)
        prev_small = small
        idx += 1
    cap.release()
    return magnitudes


def summarize_motion(magnitudes: List[float], min_magnitude: float = 2.0, diag_units: bool = True) -> dict:
    if not magnitudes:
        return {
            "mean_magnitude": None, "median_magnitude": None, "std_magnitude": None,
            "p90_magnitude": None, "proportion_above_threshold": None, "samples": 0,
            "units": "normalised optical-flow magnitude (0-1)",
            "note": "No frames sampled (empty or unreadable video).",
        }
    arr = np.array(magnitudes, dtype=np.float64)
    threshold = min_magnitude / 256.0 if diag_units else min_magnitude
    return {
        "mean_magnitude": round(float(arr.mean()), 6),
        "median_magnitude": round(float(np.median(arr)), 6),
        "std_magnitude": round(float(arr.std()), 6),
        "p90_magnitude": round(float(np.percentile(arr, 90)), 6),
        "proportion_above_threshold": round(float(np.mean(arr > threshold)), 4),
        "samples": int(len(arr)),
        "units": "normalised optical-flow magnitude (0-1)",
        "note": (
            "Optical flow conflates camera motion with subject motion; these "
            "metrics do NOT isolate movement of depicted subjects."
        ),
    }


def analyze_motion(
    path: str, frame_interval_seconds: float = 1.0, block_size: int = 16,
    min_magnitude: float = 2.0, max_frames: int = 0
) -> dict:
    mags = sample_motion(path, frame_interval_seconds, block_size, min_magnitude, max_frames)
    return summarize_motion(mags, min_magnitude)
