"""Video metadata extraction via OpenCV.

Recorded: duration, frame rate, frame count, resolution, aspect ratio, codec
(when available), and audio presence as boolean metadata only.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2

from ..core.schemas import VideoMetadata

logger = logging.getLogger("artwork_analyzer.video.metadata")


def _fourcc_to_string(code: int) -> str:
    if code is None or code == 0:
        return ""
    chars = [(code >> 8 * i) & 0xFF for i in range(4)]
    try:
        return "".join(chr(c) for c in chars)
    except ValueError:
        return str(code)


def probe_video(path: str) -> VideoMetadata:
    """Probe a video file and return its metadata."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    # Frame count is unreliable for some containers; also derive duration.
    duration = frame_count / fps if fps > 0 else None
    if duration is None:
        # Fallback: read to end.
        last_ts = 0.0
        n = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            last_ts = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
            n += 1
        frame_count = max(frame_count, n)
        fps = n / last_ts if last_ts > 0 else fps
        duration = last_ts if last_ts else None

    aspect_ratio = round(width / height, 4) if height else None

    # Audio presence: OpenCV cannot reliably detect audio tracks. We report
    # it as unknown rather than guessing. (A ffmpeg probe would be needed.)
    has_audio: Optional[bool] = None

    cap.release()
    return VideoMetadata(
        duration_seconds=duration,
        frame_rate=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        codec=_fourcc_to_string(fourcc) if fourcc else None,
        has_audio=has_audio,
    )
