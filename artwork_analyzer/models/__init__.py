"""Detector backends and model registry."""

from .registry import REGISTRY, spec, resolve_model, model_report, ModelSpec
from .detectors import (
    RawDetection, DetectionBackend, YuNetBackend,
    ORTSSDv1Backend, ORTYOLOv8Backend, ORTCVBackend,
    VOC_LABELS, COCO_LABELS, COCO_PERSON_LABEL, _int_dim,
)

__all__ = [
    "REGISTRY", "spec", "resolve_model", "model_report", "ModelSpec",
    "RawDetection", "DetectionBackend", "YuNetBackend",
    "ORTSSDv1Backend", "ORTYOLOv8Backend", "ORTCVBackend",
    "VOC_LABELS", "COCO_LABELS", "COCO_PERSON_LABEL",
]
