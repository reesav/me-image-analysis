"""Tests for image I/O, EXIF handling, and metadata extraction."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from artwork_analyzer.analysis.image_io import load_image, relative_luminance


def test_load_image_metadata(tmp_path, two_rectangles):
    p = tmp_path / "r.png"
    cv2.imwrite(str(p), cv2.cvtColor(two_rectangles, cv2.COLOR_RGB2BGR))
    res = load_image(str(p))
    assert res.width == 200 and res.height == 200
    assert res.channels == 3
    assert res.mode == "RGB"
    assert res.file_format == "PNG"
    assert res.file_size_bytes > 0
    assert abs(res.width / res.height - 1.0) < 1e-9
    np.testing.assert_array_equal(res.raw_array, res.display_array)
    assert res.exif_orientation is None
    assert "RGB" in str(res.transformations)


def _pil_with_orientation(arr_bgr, orientation, path):
    img = Image.fromarray(arr_bgr)
    exif = img.getexif()
    exif[274] = orientation
    img.save(str(path), "TIFF", exif=exif.tobytes())
    return path


def test_exif_orientation_applied(tmp_path):
    # Non-square image; orientation 6 rotates 90 CW -> swaps W and H.
    non_square = np.zeros((120, 200, 3), dtype=np.uint8)
    non_square[30:90, 50:150, 0] = 200
    bgr = cv2.cvtColor(non_square, cv2.COLOR_RGB2BGR)
    arr_bgr = non_square  # already RGB; build BGR for PIL
    bgr_for_pil = cv2.cvtColor(non_square, cv2.COLOR_RGB2BGR)
    p = _pil_with_orientation(bgr_for_pil, 6, tmp_path / "orient.tiff")
    res = load_image(str(p))
    assert res.exif_orientation == 6
    # Original 200 wide x 120 tall; after 90CW the stored "width"/height from
    # PIL is reported as-decoded, but our corrected display array is transposed.
    assert res.display_array.shape[1] == 120  # width now 120 (was height)
    assert res.display_array.shape[0] == 200  # height now 200 (was width)
    assert "EXIF orientation" in str(res.transformations)


def test_luminance_range(two_rectangles):
    lum = relative_luminance(two_rectangles)
    assert lum.min() == pytest.approx(0.0, abs=1e-6)
    assert 0.0 <= lum.max() <= 1.0


def test_load_corrupt_image_raises(tmp_path):
    p = tmp_path / "bad.png"
    p.write_bytes(b"not a real png")
    with pytest.raises(Exception):
        load_image(str(p))
