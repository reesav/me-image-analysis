"""Pydantic schemas for analysis results.

These typed models are the contract between analysis modules and storage /
export layers. They are serialisable (JSON) so that full reproducibility
records can be persisted per stimulus. Deterministic pixel measurements and
model-based predictions are kept in separate sections of each record so that
they can be distinguished.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ReviewStatus(str, Enum):
    """Status of human review for a model-based prediction."""

    AUTOMATED = "automated"
    REVIEWED_ACCEPTED = "reviewed_accepted"
    REVIEWED_MODIFIED = "reviewed_modified"
    REJECTED = "rejected"


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class ColorSpace(str, Enum):
    RGB = "rgb"
    HSV = "hsv"
    LAB = "lab"
    HSL = "hsl"


class ColorMetrics(BaseModel):
    """Deterministic colour measurements.

    Hue is aggregated as a circular quantity (mean resultant vector) rather
    than an arithmetic mean. All RGB channels are 0-255; HSV uses 0-360 deg,
    0-1 sat, 0-1 value; luminance is relative (0-1).
    """

    mean_rgb: List[float] = Field(default_factory=list, description="Mean R,G,B 0-255")
    median_rgb: List[float] = Field(default_factory=list, description="Median R,G,B 0-255")
    mean_hsv: List[float] = Field(default_factory=list, description="Mean H(0-360),S,V (0-1)")
    median_hsv: List[float] = Field(default_factory=list, description="Median H(0-360),S,V (0-1)")
    mean_luminance: float = Field(default=0.0, description="Relative luminance mean (0-1)")
    median_luminance: float = Field(default=0.0, description="Relative luminance median (0-1)")
    std_luminance: float = Field(default=0.0, description="Relative luminance std dev (0-1)")
    min_luminance: float = Field(default=0.0, description="Relative luminance minimum (0-1)")
    max_luminance: float = Field(default=0.0, description="Relative luminance maximum (0-1)")
    # Percentiles (0-1).
    p25_luminance: float = 0.0
    p75_luminance: float = 0.0
    p10_luminance: float = 0.0
    p90_luminance: float = 0.0
    hue_circular_mean: Optional[float] = Field(
        default=None, description="Circular mean hue in degrees (0-360)"
    )
    colorfulness: Optional[float] = Field(
        default=None,
        description="Hasler-Susstrunk perceived colourfulness metric",
    )
    channel_histograms: Optional[dict] = Field(
        default=None,
        description="Per-channel 256-bin histograms (R,G,B) as lists",
    )
    dominant_colors: Optional[List[List[float]]] = Field(
        default=None, description="Top-N dominant RGB colours 0-255"
    )
    color_proportions: Optional[dict] = Field(
        default=None,
        description="Proportion of pixels in configurable colour ranges",
    )


class BrightnessContrastMetrics(BaseModel):
    """Clearly-named brightness and contrast metrics.

    ``rms_contrast`` is the root-mean-square of normalised intensity
    (Reinhard definition). ``michelson_contrast`` is only reported when its
    assumption (a region of dark and a region of bright foreground) holds,
    signalled via ``michelson_applicable``.
    """

    mean_luminance: float = 0.0
    median_luminance: float = 0.0
    std_luminance: float = 0.0
    min_luminance: float = 0.0
    max_luminance: float = 0.0
    p10_luminance: float = 0.0
    p25_luminance: float = 0.0
    p75_luminance: float = 0.0
    p90_luminance: float = 0.0
    rms_contrast: Optional[float] = None
    michelson_contrast: Optional[float] = None
    michelson_applicable: bool = False
    local_contrast_mean: Optional[float] = None
    local_contrast_std: Optional[float] = None


class CompositionMetrics(BaseModel):
    """Deterministic composition and spatial-distribution metrics.

    Foreground masks are estimated with a lightweight saliency/threshold
    method when segmentation is not available, so these are best-effort
    proxies. Names are explicit about the underlying technique.
    """

    horizontal_foreground_distribution: Optional[List[float]] = None
    vertical_foreground_distribution: Optional[List[float]] = None
    grid_occupancy: Optional[List[List[float]]] = None
    center_of_mass_x: Optional[float] = None
    center_of_mass_y: Optional[float] = None
    foreground_background_ratio: Optional[float] = None
    rule_of_thirds_alignment: Optional[List[float]] = None


class DetectionBox(BaseModel):
    """Normalised bounding box: x,y,width,height in 0-1 image coordinates."""

    x: float
    y: float
    width: float
    height: float
    confidence: float
    label: str = "person"
    review_status: ReviewStatus = ReviewStatus.AUTOMATED
    review_note: Optional[str] = None

    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)


class PersonRecord(BaseModel):
    """A single detected or reviewed person figure."""

    person_id: str
    box: Optional[DetectionBox] = None
    box_area_percent: float = 0.0
    # Segmentation (may be None if not produced/available).
    mask_area_pixels: Optional[float] = None
    mask_area_percent: Optional[float] = None
    mask_available: bool = False
    mask_confidence: Optional[float] = None
    centroid_x: Optional[float] = None
    centroid_y: Optional[float] = None
    review_status: ReviewStatus = ReviewStatus.AUTOMATED
    review_note: Optional[str] = None
    detection_model: str = ""
    segmentation_model: str = ""


class FaceRecord(BaseModel):
    """A single detected face."""

    face_id: str
    box: DetectionBox
    box_area_percent: float = 0.0
    is_cropped: bool = False
    detection_model: str = ""
    review_status: ReviewStatus = ReviewStatus.AUTOMATED
    review_note: Optional[str] = None


class ObjectRecord(BaseModel):
    """A generic detected object/category (optional content analysis)."""

    object_id: str
    label: str
    confidence: float
    box: DetectionBox
    review_status: ReviewStatus = ReviewStatus.AUTOMATED


class VideoMetadata(BaseModel):
    duration_seconds: Optional[float] = None
    frame_rate: Optional[float] = None
    frame_count: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    aspect_ratio: Optional[float] = None
    codec: Optional[str] = None
    has_audio: Optional[bool] = None


class SceneCut(BaseModel):
    type: str  # "hard" or "gradual"
    timestamp_seconds: float
    frame_index: int


class MotionMetrics(BaseModel):
    mean_magnitude: Optional[float] = None
    median_magnitude: Optional[float] = None
    std_magnitude: Optional[float] = None
    p90_magnitude: Optional[float] = None
    proportion_above_threshold: Optional[float] = None
    samples: int = 0
    units: str = "normalised optical-flow magnitude (0-1)"


class FrameSample(BaseModel):
    frame_index: int
    timestamp_seconds: float
    color: Optional[ColorMetrics] = None
    brightness: Optional[BrightnessContrastMetrics] = None
    person_count: Optional[int] = None
    face_count: Optional[int] = None
    person_area_percent: Optional[float] = None
    segmentation_area_percent: Optional[Optional[float]] = None


class Reproducibility(BaseModel):
    app_version: str = ""
    analysis_module_versions: dict = Field(default_factory=dict)
    person_detector_model: str = ""
    person_detector_version: str = ""
    face_detector_model: str = ""
    segmentation_model: str = ""
    segmentation_backend: str = ""
    confidence_thresholds: dict = Field(default_factory=dict)
    video_frame_interval_seconds: Optional[float] = None
    scene_cut_threshold: Optional[float] = None
    device: str = "cpu"
    preprocessing: str = ""
    colour_space: str = "sRGB relative luminance"
    processing_timestamp: str = ""
    dependency_versions: dict = Field(default_factory=dict)


class StimulusAnalysis(BaseModel):
    """Aggregate per-stimulus analysis result."""

    stimulus_id: str
    file_path: str
    file_name: str
    media_type: MediaType = MediaType.IMAGE
    analysis_version: str = "1.0.0"
    image_metadata: Optional[dict] = None
    color: Optional[ColorMetrics] = None
    brightness_contrast: Optional[BrightnessContrastMetrics] = None
    composition: Optional[CompositionMetrics] = None
    people: List[PersonRecord] = Field(default_factory=list)
    faces: List[FaceRecord] = Field(default_factory=list)
    objects: List[ObjectRecord] = Field(default_factory=list)
    video_metadata: Optional[VideoMetadata] = None
    scene_cuts: List[SceneCut] = Field(default_factory=list)
    motion: Optional[MotionMetrics] = None
    frame_samples: List[FrameSample] = Field(default_factory=list)
    reproducibility: Reproducibility = Field(default_factory=Reproducibility)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    completed: bool = False
    review_status: ReviewStatus = ReviewStatus.REVIEWED_ACCEPTED
    correction_history: List[dict] = Field(default_factory=list)
