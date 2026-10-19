"""Model management: lazy loading, caching, and licence metadata.

Model weights are NEVER checked into the package source tree except where
explicitly permitted by their licence. The small face-detection model
(YuNet, Apache-2.0) is bundled as package data for zero-download first-run
face detection. Heavier models (SSD-MobileNet-v1, YOLOv8n) are downloaded
on first use into the user model cache and are documented with their
approximate size and licence.

Default detector stack (CPU, no torch required):
  faces  -> face_yunet  (cv2.FaceDetectorYN_create, bundled ONNX, Apache-2.0, 0.2 MB)
  persons/objects -> onnx_ssd_mobilenet_v1 (onnxruntime, Apache-2.0, ~28 MB, COCO/VOC person=class 15)

Heavier model plugins (require torch + extra installs) are documented but not
the default.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from urllib.request import urlretrieve

from ..core.paths import model_cache_dir

logger = logging.getLogger("artwork_analyzer.models.registry")

PACKAGE_DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    source: str  # "bundled" | "download"
    local_name: str
    url: str = ""
    approx_size_mb: float = 0.0
    license: str = ""
    licence_notes: str = ""


REGISTRY: Dict[str, ModelSpec] = {
    "face_yunet": ModelSpec(
        name="face_yunet",
        description="OpenCV YuNet face detector (frontal + reasonable profile), bundled",
        source="bundled",
        local_name="face_detection_yunet_2023mar.onnx",
        approx_size_mb=0.23,
        license="Apache-2.0 (OpenCV)",
        licence_notes="Bundled package data (redistributable, permissive).",
    ),
    "onnx_ssd_mobilenet_v1": ModelSpec(
        name="onnx_ssd_mobilenet_v1",
        description="SSD-MobileNet-v1 ONNX (VOC 20), COCO 'person'=class 15",
        source="download",
        local_name="ssd_mobilenet_v1.onnx",
        url="https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_10.onnx",
        approx_size_mb=28.0,
        license="Apache-2.0 (ONNX Model Zoo)",
        licence_notes="Downloaded at runtime into the user cache; not redistributed.",
    ),
    "onnx_yolov8n": ModelSpec(
        name="onnx_yolov8n",
        description="YOLOv8n ONNX (COCO, 'person'=class 0), via onnxruntime",
        source="download",
        local_name="yolov8n.onnx",
        url="https://github.com/onnx/models/raw/main/validated/vision/body_analysis/yolov8/yolov8n.onnx",
        approx_size_mb=6.2,
        license="AGPL-3.0-or-later (Ultralytics)",
        licence_notes="Downloaded at runtime; AGPL-3.0 applies to network-service redistribution; local use permitted.",
    ),
}


def spec(name: str) -> Optional[ModelSpec]:
    return REGISTRY.get(name)


def bundled_path(spec_obj: ModelSpec) -> Path:
    return PACKAGE_DATA_DIR / spec_obj.local_name


def cached_path(spec_obj: ModelSpec) -> Path:
    return model_cache_dir() / spec_obj.local_name


def resolve_model(name: str) -> Optional[Path]:
    """Return an existing local path for a model, downloading if required.

    Returns None if the model cannot be resolved (e.g. download failed).
    """
    sp = REGISTRY.get(name)
    if sp is None:
        logger.error("Unknown model requested: %s", name)
        return None
    if sp.source == "bundled":
        return bundled_path(sp)
    cache = cached_path(sp)
    if cache.exists() and cache.stat().st_size > 0:
        return cache
    logger.info(
        "Downloading model %s (%.1f MB, %s) to %s",
        sp.name, sp.approx_size_mb, sp.license, cache,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp = cache.with_suffix(cache.suffix + ".part")
        urlretrieve(sp.url, tmp)
        tmp.replace(cache)
        return cache
    except Exception as exc:
        logger.warning("Failed to download model %s: %s", sp.name, exc)
        return None


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def model_report() -> Dict[str, dict]:
    """Return a licence/availability report of all registered models."""
    report: Dict[str, dict] = {}
    for key, sp in REGISTRY.items():
        if sp.source == "bundled":
            available = bundled_path(sp).exists()
        else:
            available = cached_path(sp).exists()
        report[key] = {
            "description": sp.description,
            "source": sp.source,
            "license": sp.license,
            "licence_notes": sp.licence_notes,
            "approx_size_mb": sp.approx_size_mb,
            "url": sp.url,
            "available": available,
        }
    return report
