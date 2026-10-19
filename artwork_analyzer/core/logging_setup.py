"""Centralised logging configuration.

Provides named loggers with consistent formatting so that processing, analysis
and review operations emit useful, filterable messages. Logging is configured
explicitly (never via basicConfig side effects inside analysis modules).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_DEFAULT_FORMAT = (
    "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    capture_exceptions: bool = True,
) -> logging.RootLogger:
    """Configure and return the root logger.

    - ``level``: logging level name (e.g. ``"INFO"``, ``"DEBUG"``).
    - ``log_file``: optional path to a file handler; parent dirs created.
    - ``capture_exceptions``: install ``exception`` hook that logs uncaught
      exceptions through the logging system.
    """
    level_num = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger("artwork_analyzer")
    root.setLevel(level_num)
    # Avoid duplicate handlers if called more than once.
    root.handlers.clear()

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(
        logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)
    )
    root.addHandler(stream_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)
        )
        root.addHandler(file_handler)

    if capture_exceptions:
        sys.excepthook = lambda exc_type, exc, tb: root.exception(
            "Uncaught exception", exc_info=(exc_type, exc, tb)
        )

    return root


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``artwork_analyzer`` namespace."""
    if name == "artwork_analyzer" or not name.startswith("artwork_analyzer."):
        name = f"artwork_analyzer.{name}"
    return logging.getLogger(name)
