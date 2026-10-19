"""Explicit, serialisable application settings.

Settings are persisted to a JSON file so that analyses can be reproduced.
Values are intentionally simple types so the settings file is human-readable
and diff-friendly. Environment variable overrides of the form
``AA_<FIELD>`` are applied at load time for scripting.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from .paths import user_config_dir


def _env(name: str, default: Any) -> Any:
    value = os.environ.get(f"AA_{name}")
    if value is None:
        return default
    # Coerce booleans.
    if isinstance(default, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return value


class DetectorSettings(BaseModel):
    """Model-selection and threshold settings for detection/segmentation.

    ``person_detector`` may be ``disabled``, ``haar_fullbody`` (bundled,
    weak), ``onnx_ssd`` (auto-downloaded SSD-MobileNet-v2 COCO) or an
    installed plugin name. ``segmentation_backend`` may be ``grabcut``
    (default, CPU, no download), ``mediapipe`` or ``sam2`` (optional heavy).
    """

    person_detector: str = Field(default="onnx_ssd_mobilenet_v1")
    person_confidence_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    face_detector: str = Field(default="face_yunet")
    face_confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    face_score_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    face_nms_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    segmentation_backend: str = Field(default="grabcut")
    inference_max_size: int = Field(default=1024, ge=128, le=8192)
    device: str = Field(default="cpu")

    # Whether to run optional object/scene category detection (model-based).
    object_detection_enabled: bool = Field(default=False)
    object_detector: str = Field(default="onnx_ssd_mobilenet_v1")
    object_confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)


class VideoSettings(BaseModel):
    """Video processing settings that affect temporal resolution and metrics."""

    frame_interval_seconds: float = Field(default=1.0, ge=0.01)
    scene_cut_threshold: float = Field(default=27.0, ge=0.0, le=255.0)
    motion_block_size: int = Field(default=16, ge=4, le=128)
    motion_min_magnitude: float = Field(default=2.0, ge=0.0)
    downscale_for_motion: bool = Field(default=True)


class ProcessingSettings(BaseModel):
    parallel_workers: int = Field(default=1, ge=1)
    memory_limit_mb: int = Field(default=4096, ge=256)
    skip_completed: bool = Field(default=True)
    video_sample_people: bool = Field(default=False)


class ExportSettings(BaseModel):
    default_dir: Optional[str] = None
    include_reviewed: bool = Field(default=True)
    include_automated: bool = Field(default=True)


class AppSettings(BaseModel):
    """Top-level application settings persisted to disk."""

    app_version: str = "1.0.0"
    detector: DetectorSettings = Field(default_factory=DetectorSettings)
    video: VideoSettings = Field(default_factory=VideoSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)


def default_config_path() -> Path:
    return user_config_dir() / "settings.json"


def load_settings(path: Optional[Path] = None) -> AppSettings:
    target = Path(path) if path else default_config_path()
    if target.exists():
        data = json.loads(target.read_text(encoding="utf-8"))
    else:
        data = {}
    # Apply environment overrides.
    env_mapping: Dict[str, str] = {
        "PERSON_DETECTOR": "detector.person_detector",
        "FACE_DETECTOR": "detector.face_detector",
        "SEGMENTATION_BACKEND": "segmentation_backend",
        "DEVICE": "detector.device",
    }
    for env_name, field in env_mapping.items():
        if os.environ.get(f"AA_{env_name}") is not None:
            data.setdefault("detector" if "." not in field else {}, {})
    settings = AppSettings.model_validate(data)
    return settings


def save_settings(settings: AppSettings, path: Optional[Path] = None) -> Path:
    target = Path(path) if path else default_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
    return target
