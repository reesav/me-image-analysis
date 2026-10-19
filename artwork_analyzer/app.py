"""Console entry point: launches the Streamlit dashboard.

Usage:
    artwork-analyzer            # runs `streamlit run artwork_analyzer/ui/app.py`
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    ui_app = Path(__file__).resolve().parent / "ui" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(ui_app)]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    raise SystemExit(main())
