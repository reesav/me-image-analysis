"""Image loading and metadata extraction.

Responsibilities:
- Decode an image without ever modifying the original file.
- Extract EXIF orientation / metadata and embedded colour profile.
- Provide both the *raw* decoded array and the *orientation-corrected*
  array used for analysis. All corrections are documented in the returned
  metadata so transformations are auditable.

Colour convention: arrays are returned in **RGB** order (uint8). Luminance is
the sRGB relative luminance per ITU-R BT.709.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ExifTags

logger = logging.getLogger("artwork_analyzer.analysis.image_io")


# Standard sRGB relative-luminance weights (ITU-R BT.709).
LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)


@dataclass
class ImageResult:
    """Container for a decoded image plus its metadata."""

    file_path: str
    file_name: str
    relative_path: str
    stimulus_id: str
    raw_array: np.ndarray  # RGB uint8, as decoded (no orientation fix)
    display_array: np.ndarray  # RGB uint8, orientation-corrected for analysis
    width: int
    height: int
    channels: int
    mode: str  # PIL mode, e.g. "RGB", "RGBA", "P", "L"
    file_size_bytes: int
    file_format: str
    color_profile: Optional[str]
    exif_orientation: Optional[int]
    exif_metadata: Dict[str, Any]
    transformations: list = field(default_factory=list)
    decoding_incomplete: bool = False
    warnings: list = field(default_factory=list)


def _pillow_metadata(pil_img: Image.Image) -> Tuple[Optional[str], Optional[int], Dict[str, Any]]:
    """Extract colour profile and EXIF metadata from a Pillow image.

    Must be called BEFORE ``pil_img.load()`` because loading a TIFF image
    discards the orientation tag from the merged EXIF view. Orientation is
    therefore read from the TIFF ``tag_v2`` IFD when available, falling back
    to the EXIF ``Orientation`` tag.
    """
    color_profile = None
    if "icc_profile" in pil_img.info:
        try:
            from PIL import ImageCms

            profile = ImageCms.ImageCmsProfile(pil_img.info["icc_profile"])
            color_profile = profile.profile.profile_description or "ICC (embedded)"
        except Exception:
            color_profile = "ICC (embedded, unreadable)"

    orientation = None
    # TIFF IFD first (robust before load()).
    if hasattr(pil_img, "tag_v2"):
        try:
            val = pil_img.tag_v2.get(274)
            if val is not None:
                orientation = int(val)
        except Exception:
            orientation = None
    exif_meta: Dict[str, Any] = {}
    try:
        exif = pil_img.getexif()
        if exif:
            for tag_id, value in exif.items():
                name = ExifTags.TAGS.get(tag_id, tag_id)
                # Only store JSON-serialisable values.
                try:
                    json.dumps(value)
                    exif_meta[name] = value
                except (TypeError, ValueError):
                    exif_meta[name] = str(value)
            if "Orientation" in exif_meta and orientation is None:
                orientation = int(exif_meta["Orientation"])
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("EXIF read failed for %s: %s", pil_img, exc)
    return color_profile, orientation, exif_meta


def _mode_channels(mode: str) -> int:
    return {"1": 1, "L": 1, "P": 1, "I": 1, "F": 1, "RGB": 3, "RGBA": 4,
            "RGBX": 4, "RGBa": 4, "CMYK": 4, "YCbCr": 3, "LAB": 3, "HSV": 3}.get(mode, 3)


def load_image(
    file_path: str, stimulus_id: Optional[str] = None
) -> ImageResult:
    """Load an image file and its metadata without modifying the original.

    The ``stimulus_id`` is used as a stable identifier; if omitted it is
    derived from the file stem.
    """
    path = Path(file_path)
    file_name = path.name
    relative_path = str(path)

    # Read bytes once; decode via Pillow (deterministic EXIF orientation).
    raw_bytes = path.read_bytes()
    file_size = len(raw_bytes)

    # ---- Metadata + pixels via Pillow. ----
    # Pillow does NOT auto-apply EXIF orientation on load; we apply it exactly
    # once with ImageOps.exif_transpose for the display array and keep the raw
    # decoded array untouched as the provenance record.
    from PIL import ImageOps

    pil_img = Image.open(io.BytesIO(raw_bytes))
    color_profile, orientation, exif_meta = _pillow_metadata(pil_img)
    pil_mode = pil_img.mode

    # Channel count for the stored pixel mode (for reporting only).
    channels = _mode_channels(pil_mode)

    # Raw (orientation-unaware) RGB array.
    rgb = np.array(pil_img.convert("RGB"))
    # Orientation-corrected array for analysis.
    display_rgb = np.array(ImageOps.exif_transpose(pil_img).convert("RGB"))

    transformations = ["decoded to RGB via Pillow"]
    if orientation:
        transformations.append(f"EXIF orientation {orientation}: applied once via ImageOps.exif_transpose")
    if pil_mode in ("RGBA", "LA"):
        transformations.append(f"dropped alpha (mode {pil_mode}) -> RGB")
    if pil_mode == "P":
        transformations.append(f"palette (mode P) -> RGB via Pillow")

    mode = pil_mode
    width, height = pil_img.size
    # Flag if the decoded RGB array dimensions don't match the reported size.
    decoding_incomplete = bool(rgb.shape[0] != height or rgb.shape[1] != width)

    if stimulus_id is None:
        stimulus_id = _derive_stimulus_id(file_name)

    return ImageResult(
        file_path=str(path),
        file_name=file_name,
        relative_path=relative_path,
        stimulus_id=stimulus_id,
        raw_array=rgb,
        display_array=display_rgb,
        width=width,
        height=height,
        channels=channels,
        mode=mode,
        file_size_bytes=file_size,
        file_format=pil_img.format or "",
        color_profile=color_profile,
        exif_orientation=orientation,
        exif_metadata=exif_meta,
        transformations=transformations,
        decoding_incomplete=decoding_incomplete,
    )


def _derive_stimulus_id(file_name: str) -> str:
    return Path(file_name).stem


def relative_luminance(image_rgb: np.ndarray) -> np.ndarray:
    """Compute relative luminance (0-1) for an RGB uint8 array (HxWx3)."""
    rgb = image_rgb.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return LUMA_WEIGHTS[0] * r + LUMA_WEIGHTS[1] * g + LUMA_WEIGHTS[2] * b


def rgb_to_hsv(image_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 image to HSV float image (H 0-360, S,V 0-1)."""
    rgb = image_rgb.astype(np.float32) / 255.0
    hsv = np.zeros(rgb.shape, dtype=np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=2)
    mn = np.min(rgb, axis=2)
    diff = mx - mn
    # Value
    hsv[..., 2] = mx
    # Saturation
    with np.errstate(divide="ignore", invalid="ignore"):
        sat = np.where(mx > 0, diff / mx, 0.0)
    hsv[..., 1] = sat
    # Hue
    h = np.zeros_like(mx)
    mask = diff > 0
    # Red is max
    idx = mask & (mx == r)
    h[idx] = ((g[idx] - b[idx]) / diff[idx]) % 6.0
    # Green is max
    idx = mask & (mx == g)
    h[idx] = (b[idx] - r[idx]) / diff[idx] + 2.0
    # Blue is max
    idx = mask & (mx == b)
    h[idx] = (r[idx] - g[idx]) / diff[idx] + 4.0
    h = np.where(diff > 0, h * 60.0, 0.0)
    hsv[..., 0] = h
    return hsv


def circular_mean_hue(hue_deg: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[float]:
    """Circular mean of hue in degrees (0-360), accounting for wraparound."""
    if mask is not None:
        hue_deg = hue_deg[mask]
    if hue_deg.size == 0:
        return None
    rad = np.deg2rad(hue_deg[mask] if mask is not None else hue_deg)
    sin_sum = np.mean(np.sin(rad))
    cos_sum = np.mean(np.cos(rad))
    if sin_sum == 0.0 and cos_sum == 0.0:
        return 0.0
    mean_rad = np.arctan2(sin_sum, cos_sum)
    if mean_rad < 0:
        mean_rad += 2 * np.pi
    deg = float(np.rad2deg(mean_rad))
    # Snap values that are within float-epsilon of 360 back to 0 (a true mean
    # of 0 degrees and 360 degrees are the same direction).
    if abs(deg - 360.0) < 1e-6:
        return 0.0
    return deg
