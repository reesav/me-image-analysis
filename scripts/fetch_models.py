"""Fetch small model/weight files used by the default backends.

The Haar cascade XMLs are tiny (OpenCV, BSD-3) and are bundled into the
package data directory so that face/person detection works with zero runtime
downloads. Heavier ONNX models (YOLOv8n, SSD-MobileNet v2) are optional and
are also downloaded here into the user model cache.

Run:  python scripts/fetch_models.py [--with-onnx]

Approximate sizes:
  haar_frontalface_default.xml      ~0.9 MB  (bundled)
  haarcascade_fullbody.xml          ~2.0 MB  (bundled)
  haarcascade_upperbody.xml         ~1.0 MB  (bundled)
  yolov8n.onnx (optional)           ~6.2 MB  (AGPL-3, runtime cache)
  ssd_mobilenet_v2.onnx (optional) ~13.0 MB  (Apache-2.0, runtime cache)
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from artwork_analyzer.core.logging_setup import configure_logging, get_logger
from artwork_analyzer.models.registry import REGISTRY, PACKAGE_DATA_DIR, resolve_model

log = get_logger("scripts.fetch_models")


def fetch_bundled() -> None:
    PACKAGE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    urls = {
        "haar_frontalface": "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml",
        "haar_frontalface_alt2": "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_alt2.xml",
        "haar_profileface": "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_profileface.xml",
        "haar_fullbody": "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_fullbody.xml",
        "haar_upperbody": "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_upperbody.xml",
    }
    # The registry already has URLs? No - bundled ones have empty url. Add here.
    import urllib.request
    for key in ("haar_frontalface", "haar_frontalface_alt2", "haar_profileface", "haar_fullbody", "haar_upperbody"):
        spec = REGISTRY[key]
        dest = PACKAGE_DATA_DIR / spec.local_name
        if dest.exists() and dest.stat().st_size > 100:
            log.info("Already present: %s (%d bytes)", spec.local_name, dest.stat().st_size)
            continue
        url = urls[key]
        log.info("Downloading %s (~%.1f MB, %s)", spec.local_name, spec.approx_size_mb, spec.license)
        try:
            urllib.request.urlretrieve(url, dest)
            log.info("Saved %s", dest)
        except Exception as exc:
            log.warning("Failed to fetch %s: %s", spec.local_name, exc)


def fetch_optional() -> None:
    for key in ("onnx_yolov8n", "onnx_ssd_mobilenet_v2"):
        spec = REGISTRY[key]
        log.info("Fetching optional model %s (~%.1f MB, %s)", spec.name, spec.approx_size_mb, spec.license)
        path = resolve_model(key)
        if path is None:
            log.warning("Could not resolve %s", key)
        else:
            log.info("Saved optional model to %s", path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch model weight files")
    ap.add_argument("--with-onnx", action="store_true", help="Also download optional ONNX detectors")
    args = ap.parse_args()
    configure_logging("INFO")
    fetch_bundled()
    if args.with_onnx:
        fetch_optional()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
