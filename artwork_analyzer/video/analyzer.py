"""Video analysis orchestrator: metadata + scene cuts + motion + frame samples.

Frame sampling is configurable to trade speed vs temporal resolution. Per-frame
person/face detection is opt-in because it is the most expensive step.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import cv2
import numpy as np

from ..core.config import AppSettings
from ..core.schemas import (
    StimulusAnalysis, Reproducibility, MediaType, SceneCut, FrameSample,
)
from ..core.version import __version__
from .metadata import probe_video
from .scene_detection import detect_scene_cuts
from .motion import analyze_motion
from ..analysis.color import analyze_color
from ..analysis.brightness_contrast import analyze_brightness_contrast

logger = logging.getLogger("artwork_analyzer.video.analyzer")


def _sample_frame_indices(frame_count: int, fps: float, interval: float) -> List[int]:
    if fps <= 0 or frame_count <= 0:
        return []
    step = max(1, int(round(interval * fps)))
    return list(range(0, frame_count, step))


def analyze_video(path: str, settings: AppSettings, stimulus_id: Optional[str] = None) -> StimulusAnalysis:
    from ..analysis.image_io import load_image  # for id derivation
    if stimulus_id is None:
        stimulus_id = path

    t0 = time.time()
    result = StimulusAnalysis(
        stimulus_id=stimulus_id,
        file_path=path,
        file_name=path.split("/")[-1],
        media_type=MediaType.VIDEO,
    )
    try:
        meta = probe_video(path)
        result.video_metadata = meta
    except Exception as exc:
        result.errors.append(f"video_probe_failed: {exc}")
        logger.exception("Video probe failed for %s", path)
        result.reproducibility = Reproducibility(
            app_version=__version__,
            analysis_module_versions={},
            processing_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        return result

    fps = meta.frame_rate or 1.0
    frame_count = meta.frame_count or 0

    # Scene cuts
    try:
        cuts = detect_scene_cuts(
            path,
            threshold=settings.video.scene_cut_threshold,
            max_frames=0,
        )
        result.scene_cuts = cuts
    except Exception as exc:
        result.errors.append(f"scene_detection_failed: {exc}")

    # Motion
    try:
        metrics_dict = analyze_motion(
            path,
            frame_interval_seconds=settings.video.frame_interval_seconds,
            block_size=settings.video.motion_block_size,
            min_magnitude=settings.video.motion_min_magnitude,
        )
        from ..core.schemas import MotionMetrics
        result.motion = MotionMetrics(
            mean_magnitude=metrics_dict["mean_magnitude"],
            median_magnitude=metrics_dict["median_magnitude"],
            std_magnitude=metrics_dict["std_magnitude"],
            p90_magnitude=metrics_dict["p90_magnitude"],
            proportion_above_threshold=metrics_dict["proportion_above_threshold"],
            samples=metrics_dict["samples"],
            units=metrics_dict["units"],
        )
        motion_metrics = metrics_dict
    except Exception as exc:
        result.errors.append(f"motion_analysis_failed: {exc}")

    # Frame sampling
    try:
        indices = _sample_frame_indices(frame_count, fps, settings.video.frame_interval_seconds)
        sample_people = settings.processing.video_sample_people
        frames: List[FrameSample] = []
        cap = cv2.VideoCapture(path)
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = (fi / fps) if fps else 0.0
            color = analyze_color(rgb)
            bc = analyze_brightness_contrast(rgb)
            people_count: Optional[int] = None
            face_count: Optional[int] = None
            person_area: Optional[float] = None
            if sample_people:
                from ..analysis.person_detection import detect_people
                from ..analysis.face_detection import detect_faces
                from ..analysis.segmentation import segment_people, apply_masks
                people, _, _, ok = detect_people(rgb, settings.detector)
                faces, _, _ = detect_faces(rgb, settings.detector)
                people_count = len(people)
                face_count = len(faces)
                if ok and people:
                    masks = segment_people(rgb, people, settings.detector)
                    apply_masks(people, masks)
                    areas = [p.mask_area_percent for p in people if p.mask_area_percent is not None]
                    if areas:
                        person_area = round(float(np.mean(areas)), 4)
            frames.append(
                FrameSample(
                    frame_index=fi,
                    timestamp_seconds=round(ts, 3),
                    color=color,
                    brightness=bc,
                    person_count=people_count,
                    face_count=face_count,
                    person_area_percent=person_area,
                )
            )
        cap.release()
        result.frame_samples = frames
    except Exception as exc:
        result.errors.append(f"frame_sampling_failed: {exc}")
        logger.exception("Frame sampling failed for %s", path)
    if not result.errors:
        result.completed = True

    result.reproducibility = Reproducibility(
        app_version=__version__,
        analysis_module_versions={
            "video": "1.0", "scene_detection": "1.0", "motion": "1.0",
            "color": "1.0", "brightness_contrast": "1.0",
        },
        video_frame_interval_seconds=settings.video.frame_interval_seconds,
        scene_cut_threshold=settings.video.scene_cut_threshold,
        device=settings.detector.device,
        preprocessing="Frames decoded as RGB; scene cuts via Bhattacharyya histogram distance (0-255 scale)",
        colour_space="sRGB relative luminance (0-1)",
        processing_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        dependency_versions={
            k: __import__(k).__version__ for k in ("numpy", "cv2")
        },
    )
    result.reproducibility.dependency_versions["opencv"] = result.reproducibility.dependency_versions.pop("cv2")
    logger.info("Analyzed video %s in %.2fs", path, time.time() - t0)
    return result
