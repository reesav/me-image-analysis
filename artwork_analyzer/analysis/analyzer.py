"""Top-level analysis orchestrator.

Ties loading + deterministic measurements + detection + segmentation into a
:class:`StimulusAnalysis` record. Every step is timed and exceptions per-step
are caught and recorded as warnings/errors so a single failure does not abort
a whole batch. Detection/segmentation are model-based; deterministic pixel
measurements are separated in the schema.

Module versions are recorded for reproducibility; deterministic steps have a
fixed versioned definition string.
"""

from __future__ import annotations

import logging
import time
import traceback
from typing import Optional

import numpy as np

from ..core.config import AppSettings
from ..core.schemas import (
    StimulusAnalysis, Reproducibility, MediaType, ReviewStatus,
    ColorMetrics, BrightnessContrastMetrics, CompositionMetrics,
)
from ..core.version import __version__
from ..core.version import __version__
from .image_io import load_image, ImageResult
from .color import analyze_color
from .brightness_contrast import analyze_brightness_contrast
from .composition import analyze_composition
from .face_detection import detect_faces
from .person_detection import detect_people
from .segmentation import segment_people, apply_masks
from .objects import detect_objects

logger = logging.getLogger("artwork_analyzer.analysis.analyzer")

DETERMINISTIC_DEFINITIONS = {
    "color": "artwork_analyzer.analysis.color; sVD-relative luminance BT.709; hue circular mean",
    "brightness_contrast": (
        "artwork_analyzer.analysis.brightness_contrast; RMS contrast = std(luminance);"
        " Michelson (Lmax-Lmin)/(Lmax+Lmin) when bimodal"
    ),
    "composition": "artwork_analyzer.analysis.composition; blur-difference saliency foreground proxy",
}

ANALYSIS_MODULE_VERSIONS = {
    "color": "1.0",
    "brightness_contrast": "1.0",
    "composition": "1.0",
    "face_detection": "1.0",
    "person_detection": "1.0",
    "segmentation": "1.0",
    "video": "1.0",
}


def _dependency_versions() -> dict:
    import cv2, skimage, scipy, numpy
    deps = {
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
        "scikit_image": skimage.__version__,
        "scipy": scipy.__version__,
    }
    try:
        import pandas
        deps["pandas"] = pandas.__version__
    except Exception:
        pass
    try:
        import onnxruntime as ort
        deps["onnxruntime"] = ort.__version__
    except Exception:
        pass
    try:
        import streamlit
        deps["streamlit"] = streamlit.__version__
    except Exception:
        pass
    return deps


def analyze_image(
    file_path: str, settings: AppSettings, stimulus_id: Optional[str] = None
) -> StimulusAnalysis:
    """Run the full image analysis pipeline for one file."""
    result: StimulusAnalysis
    t0 = time.time()
    try:
        img: ImageResult = load_image(file_path, stimulus_id=stimulus_id)
    except Exception as exc:
        result = StimulusAnalysis(
            stimulus_id=stimulus_id or file_path,
            file_path=file_path,
            file_name=file_path,
            media_type=MediaType.IMAGE,
            errors=[f"image_load_failed: {exc}"],
        )
        logger.exception("Failed to load image %s", file_path)
        return result

    display = img.display_array
    width, height = img.width, img.height

    image_meta = {
        "file_name": img.file_name,
        "relative_path": img.relative_path,
        "file_format": img.file_format,
        "file_size_bytes": img.file_size_bytes,
        "width": width,
        "height": height,
        "pixel_count": width * height,
        "aspect_ratio": round(width / height, 4) if height else None,
        "mode": img.mode,
        "channels": img.channels,
        "color_profile": img.color_profile,
        "exif_orientation": img.exif_orientation,
        "exif_metadata": img.exif_metadata,
        "transformations_applied": img.transformations,
        "decoding_incomplete": img.decoding_incomplete,
        "stimulus_id": img.stimulus_id,
    }

    result = StimulusAnalysis(
        stimulus_id=img.stimulus_id,
        file_path=img.file_path,
        file_name=img.file_name,
        media_type=MediaType.IMAGE,
        image_metadata=image_meta,
    )

    warnings: list = []
    errors: list = []

    # --- Deterministic measurements ---
    for step, fn, model_cls in (
        ("color", analyze_color, ColorMetrics),
        ("brightness_contrast", analyze_brightness_contrast, BrightnessContrastMetrics),
        ("composition", analyze_composition, CompositionMetrics),
    ):
        try:
            value = fn(display)
            setattr(result, step, model_cls(**value))
        except Exception as exc:
            errors.append(f"{step}_failed: {exc}")
            logger.exception("Step %s failed for %s", step, file_path)

    # --- Model-based: face detection ---
    try:
        faces, fb_name, fb_model, face_available = detect_faces(display, settings.detector)
        result.faces = faces
        face_info = (fb_name, fb_model)
    except Exception as exc:
        errors.append(f"face_detection_failed: {exc}")
        face_info = ("disabled", "none")
        logger.exception("Face detection failed for %s", file_path)

    # --- Model-based: person detection ---
    people, pb_name, pb_model, person_available = detect_people(display, settings.detector)
    if not person_available:
        warnings.append("person_detection_unavailable")
    result.people = people
    person_info = (pb_name, pb_model)

    # --- Segmentation (only for detected people) ---
    seg_records = []
    masks_info = []
    try:
        if person_available and people:
            masks_info = segment_people(display, people, settings.detector)
            apply_masks(people, masks_info)
            seg_records = [
                {"person_id": pid, "backend": b, "confidence": float(c), "pixels": int(m.sum())}
                for pid, m, c, b in masks_info
            ]
    except Exception as exc:
        errors.append(f"segmentation_failed: {exc}")
        logger.exception("Segmentation failed for %s", file_path)

    # --- Optional object detection ---
    try:
        objs, ob_name, ob_model, obj_available = detect_objects(display, settings.detector)
        result.objects = objs
    except Exception as exc:
        errors.append(f"object_detection_failed: {exc}")
        obj_available = False

    result.warnings = warnings
    result.errors = errors
    if not errors:
        result.completed = True

    duration = time.time() - t0
    result.reproducibility = Reproducibility(
        app_version=__version__,
        analysis_module_versions=ANALYSIS_MODULE_VERSIONS,
        person_detector_model=person_info[1],
        person_detector_version=person_info[0],
        face_detector_model=face_info[1],
        segmentation_backend=settings.detector.segmentation_backend,
        confidence_thresholds={
            "person": settings.detector.person_confidence_threshold,
            "face": settings.detector.face_confidence_threshold,
            "object": settings.detector.object_confidence_threshold,
        },
        device=settings.detector.device,
        preprocessing="RGB decode; EXIF orientation applied to display array; channel order RGB; luminance ITU-R BT.709",
        colour_space="sRGB relative luminance (0-1); HSV H 0-360 S/V 0-1",
        processing_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        dependency_versions=_dependency_versions(),
    )
    result.reproducibility.video_frame_interval_seconds = None
    result.correction_history = []
    logger.info(
        "Analyzed %s in %.2fs (people=%d, faces=%d, masks=%d)",
        file_path, duration, len(people), len(faces), len(seg_records),
    )
    return result
