"""Streamlit UI components: overlay rendering and helper widgets.

Overlays are drawn on a copy of the RGB array with Pillow so they remain
dependency-light and testable. Masks are rendered translucent; boxes and
labels are vector-drawn on top.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from PIL import Image, ImageDraw

from artwork_analyzer.core.schemas import PersonRecord, FaceRecord, ObjectRecord


# Colour palette for figure labels.
_LABEL_COLORS = [(231, 76, 60), (46, 204, 113), (52, 152, 219), (241, 196, 14),
                 (155, 89, 187), (230, 126, 34), (149, 165, 166), (52, 152, 219)]


def _colour(i: int):
    return _LABEL_COLORS[i % len(_LABEL_COLORS)]


def _box_coords(img_w: int, img_h: int, box) -> tuple:
    x = int(box.x * img_w)
    y = int(box.y * img_h)
    bw = int(box.width * img_w)
    bh = int(box.height * img_h)
    return x, y, bw, bh


def render_overlay(
    image_rgb: np.ndarray,
    people=None,
    faces=None,
    objects=None,
    masks: Optional[Dict[str, np.ndarray]] = None,
    selected_person: Optional[str] = None,
) -> np.ndarray:
    """Return an RGB uint8 array with annotations drawn over the image."""
    people = people or []
    faces = faces or []
    objects = objects or []
    base = Image.fromarray(np.ascontiguousarray(image_rgb)).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = base.size

    # Masks (translucent).
    for i, p in enumerate(people):
        if not p.mask_available or masks is None:
            continue
        m = masks.get(p.person_id) or masks.get(i)
        if m is None or not m.any():
            continue
        col = _colour(i)
        mask_bool = (np.asarray(m) > 0)
        arr = np.array(base)
        arr[mask_bool] = (
            np.array(arr[mask_bool], dtype=np.float32) * 0.5
            + np.array(col, dtype=np.float32) * 0.5
        ).astype(np.uint8)
        # Outline via PIL on a mask image.
        alpha = np.zeros((h, w), np.uint8)
        alpha[mask_bool] = 120
        mask_img = Image.fromarray(alpha, mode="L")
        draw = ImageDraw.Draw(overlay)
        draw.rectangle([0, 0, 0, 0], fill=None)

    # Boxes + labels.
    for i, p in enumerate(people):
        x, y, bw, bh = _box_coords(w, h, p.box)
        col = _colour(i)
        draw.rectangle([x, y, x + bw, y + bh], outline=col + (255,), width=3)
        label = f"{p.person_id}" if p.box else ""
        draw.text((x + 4, y + 4), label, fill=col + (255,))

    for f in faces:
        x, y, bw, bh = _box_coords(w, h, f.box)
        draw.rectangle([x, y, x + bw, y + bh], outline=(0, 255, 255, 255), width=2)

    for i, o in enumerate(objects):
        x, y, bw, bh = _box_coords(w, h, o.box)
        draw.rectangle([x, y, x + bw, y + bh], outline=(100, 100, 100, 255), width=1)

    return np.array(Image.alpha_composite(base, overlay).convert("RGB"))


def pil_image(image_rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(np.ascontiguousarray(image_rgb))
