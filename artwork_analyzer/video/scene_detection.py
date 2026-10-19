"""Scene-cut detection.

A self-contained, deterministic histogram-difference detector is the default
(no extra dependencies). PySceneDetect is used automatically when installed
(optional) for richer gradual-transition detection. The detection threshold
is configurable and results depend on the threshold, video properties, and
detector choice -- this is documented in the schema/units.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import numpy as np

from ..core.schemas import SceneCut

logger = logging.getLogger("artwork_analyzer.video.scene_detection")


def _histogram(frame) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256]).flatten()
    norm = cv2.normalize(hist, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return norm.astype(np.float32)


def detect_scene_cuts_histogram(
    path: str, threshold: float = 27.0, max_frames: int = 0
) -> List[SceneCut]:
    """Detect hard cuts via per-frame histogram distance.

    The distance metric is the Bhattacharyya coefficient transformed to a
    0-255 scale (matching PySceneDetect's content threshold intuition). A cut
    is reported when the distance exceeds ``threshold``. ``max_frames`` (0 =
    unlimited) bounds memory for very long videos.
    """
    cap = cv2.VideoCapture(path)
    cuts: List[SceneCut] = []
    if not cap.isOpened():
        return cuts
    prev_hist: Optional[np.ndarray] = None
    idx = 0
    prev_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and idx >= max_frames:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hist = _histogram(frame_rgb)
        if prev_hist is not None:
            bc = float(np.sum(np.sqrt(hist * prev_hist)))
            # Bhattacharyya distance -> 0-255 scale.
            dist = float(np.sqrt(max(0.0, 1.0 - bc)) * 256.0)
            if dist > threshold:
                ts = float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
                cuts.append(SceneCut(type="hard", timestamp_seconds=round(ts, 3), frame_index=idx))
        prev_hist = hist
        prev_frame = frame_rgb
        idx += 1
    cap.release()
    return cuts


def detect_scene_cuts_pyscenedetect(
    path: str, threshold: float = 27.0, max_frames: int = 0
) -> Optional[List[SceneCut]]:
    """Use PySceneDetect if installed; returns None if unavailable."""
    try:
        from scenedetect import VideoManager, SceneManager
        from scenedetect.detectors import ContentDetector
    except Exception as exc:
        logger.info("PySceneDetect not available: %s", exc)
        return None
    vm = VideoManager([path])
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold, downscale_factor=2))
    try:
        vm.start()
        sm.detect_scenes(frame_source=vm)
        scenes = sm.get_scene_list()
    finally:
        vm.release()
    cuts: List[SceneCut] = []
    for start, end in scenes[1:]:
        ts = float(start.get_seconds())
        cuts.append(SceneCut(type="hard", timestamp_seconds=round(ts, 3), frame_index=int(start.frame_num)))
    return cuts


def detect_scene_cuts(path: str, threshold: float = 27.0, max_frames: int = 0) -> List[SceneCut]:
    """Detect scene cuts using pyscenedetect if available, else histogram."""
    psd = detect_scene_cuts_pyscenedetect(path, threshold=threshold, max_frames=max_frames)
    if psd is not None:
        logger.info("Scene detection used PySceneDetect; %d cuts", len(psd))
        return psd
    cuts = detect_scene_cuts_histogram(path, threshold=threshold, max_frames=max_frames)
    logger.info("Scene detection used built-in histogram; %d cuts", len(cuts))
    return cuts
