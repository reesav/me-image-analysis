"""Platform-specific path helpers without external dependencies.

Mirrors the XDG / platform conventions used by platformdirs so that model
cache, user settings and project data land in predictable locations:

    macOS   : ~/Library/Application Support/artwork_analyzer
    Linux   : ~/.local/share/artwork_analyzer
    Windows : %LOCALAPPDATA%/artwork_analyzer
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "artwork_analyzer"


def user_data_dir() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / APP_NAME
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP_NAME
    else:
        base = (
            Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
            / APP_NAME
        )
    return base


def user_config_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home())) / APP_NAME
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / APP_NAME


def model_cache_dir() -> Path:
    return user_data_dir() / "models"
