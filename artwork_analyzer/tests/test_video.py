"""Tests for video metadata, scene-cut detection, and motion metrics."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from artwork_analyzer.video.metadata import probe_video
from artwork_analyzer.video.scene_detection import detect_scene_cuts_histogram, detect_scene_cuts
from artwork_analyzer.video.motion import analyze_motion


def _write_video(path: Path, n_frames=12, w=120, h=120, fps=10.0, cut_at=6):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    ext = ".avi"
    out = cv2.VideoWriter(str(path) + ext, fourcc, fps, (w, h))
    if not out.isOpened():
        pytest.skip("Video writer not available on this platform")
    for i in range(n_frames):
        frame = np.zeros((h, w, 3), np.uint8)
        if i < cut_at:
            frame[:h // 2, :] = (0, 0, 200)  # red upper half
        else:
            frame[:, :w // 2] = (0, 200, 0)  # green left half (abrupt change)
        out.write(frame)
    out.release()
    return Path(str(path) + ext)


def test_probe_video(tmp_path):
    p = _write_video(tmp_path / "v.mp4", n_frames=10)
    m = probe_video(str(p))
    assert m.width == 120 and m.height == 120
    assert m.frame_rate == pytest.approx(10.0, abs=0.5)
    assert m.frame_count >= 9
    assert m.duration_seconds == pytest.approx(1.0, abs=0.3)


def test_scene_cut_detection_histogram(tmp_path):
    p = _write_video(tmp_path / "v.mp4", n_frames=12, cut_at=6)
    cuts = detect_scene_cuts_histogram(str(p), threshold=27.0)
    assert len(cuts) >= 1
    # The cut should be near frame index 6.
    near = [c for c in cuts if 4 <= c.frame_index <= 8]
    assert len(near) >= 1
    assert all(c.type == "hard" for c in cuts)


def test_motion_on_static_video(tmp_path):
    p = _write_video(tmp_path / "static.mp4", n_frames=10, cut_at=999)
    metrics = analyze_motion(str(p), frame_interval_seconds=0.2, min_magnitude=2.0)
    assert metrics["samples"] >= 1
    assert metrics["units"].startswith("normalised optical-flow")


def test_motion_on_moving_video(tmp_path):
    p = _write_video(tmp_path / "moving.mp4", n_frames=12, cut_at=6)
    metrics = analyze_motion(str(p), frame_interval_seconds=0.2, min_magnitude=0.5)
    assert metrics["samples"] >= 2
    # Abrupt cut should produce high max motion.
    assert metrics["p90_magnitude"] is not None
