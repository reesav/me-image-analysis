"""Tests for the Streamlit UI overlay renderer (pure-numpy, headless-safe)."""

from __future__ import annotations

import numpy as np
import pytest

from artwork_analyzer.ui.components import render_overlay, pil_image
from artwork_analyzer.core.schemas import PersonRecord, FaceRecord, DetectionBox


@pytest.fixture
def sample_image():
    arr = np.zeros((200, 200, 3), np.uint8)
    arr[:100, :100] = (200, 50, 50)  # reddish quad
    return arr


def _person(pid="person_000", x=0.1, y=0.2, w=0.2, h=0.7, conf=0.9):
    return PersonRecord(
        person_id=pid,
        box=DetectionBox(x=x, y=y, width=w, height=h, confidence=conf, label="person"),
        box_area_percent=0.0,
    )


def test_render_overlay_shape(sample_image):
    out = render_overlay(sample_image, people=[_person()])
    assert out.shape == sample_image.shape
    assert out.dtype == np.uint8


def test_render_overlay_with_masks(sample_image):
    p = _person()
    h, w = sample_image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    mask[20:160, 20:80] = 255
    out = render_overlay(sample_image, people=[p], masks={p.person_id: mask})
    # The masked region should differ from the source (overlay blended).
    diff = (out.astype(int) - sample_image.astype(int))
    assert np.abs(diff).sum() > 0


def test_render_overlay_faces_and_objects(sample_image):
    f = FaceRecord(face_id="face_000", box=DetectionBox(x=0.1, y=0.1, width=0.2, height=0.2, confidence=0.9, label="face"))
    p = _person()
    out = render_overlay(sample_image, people=[p], faces=[f])
    assert out.shape == sample_image.shape


def test_pil_image_returns_pil():
    arr = np.zeros((50, 50, 3), np.uint8)
    img = pil_image(arr)
    from PIL import Image
    assert isinstance(img, Image.Image)
    assert img.size == (50, 50)
