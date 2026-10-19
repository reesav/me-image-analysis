"""Project persistence: stimulus registry, settings, results, masks.

A project is a directory containing:
  project.json                     - settings + stimulus index
  results/<stimulus_id>.json       - one StimulusAnalysis per stimulus
  masks/<stimulus_id>/<person>.png  - reviewed/original segmentation masks
  logs/                            - processing logs per run

Stable stimulus IDs (not file paths) are used everywhere so that files can be
moved/re-named without breaking analysis records. Original predictions,
reviewed values, and correction history are stored separately inside each
result record.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.config import AppSettings, save_settings, load_settings, default_config_path
from ..core.schemas import StimulusAnalysis, MediaType, ReviewStatus
from ..core.version import __version__

logger = logging.getLogger("artwork_analyzer.storage.project")


class StimulusEntry:
    """Lightweight in-memory representation of a registered stimulus."""

    def __init__(self, stimulus_id, file_path, file_name, media_type,
                 group="", condition="", study="", source="", notes="",
                 status="pending", analysis_version=None, added_at=""):
        self.stimulus_id = stimulus_id
        self.file_path = file_path
        self.file_name = file_name
        self.media_type = media_type
        self.group = group
        self.condition = condition
        self.study = study
        self.source = source
        self.notes = notes
        self.status = status
        self.analysis_version = analysis_version
        self.added_at = added_at

    def to_dict(self) -> dict:
        return {
            "stimulus_id": self.stimulus_id,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "media_type": self.media_type,
            "group": self.group,
            "condition": self.condition,
            "study": self.study,
            "source": self.source,
            "notes": self.notes,
            "status": self.status,
            "analysis_version": self.analysis_version,
            "added_at": self.added_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StimulusEntry":
        return cls(**{k: d[k] for k in d if k in cls.__init__.__code__.co_varnames})


class Project:
    """A research project container."""

    def __init__(self, root: Path, settings: Optional[AppSettings] = None):
        self.root = Path(root)
        self.results_dir = self.root / "results"
        self.masks_dir = self.root / "masks"
        self.logs_dir = self.root / "logs"
        self.settings = settings or AppSettings()
        self.stimuli: Dict[str, StimulusEntry] = {}

    @classmethod
    def create(cls, root: Path, settings: Optional[AppSettings] = None) -> "Project":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        (root / "results").mkdir(exist_ok=True)
        (root / "masks").mkdir(exist_ok=True)
        (root / "logs").mkdir(exist_ok=True)
        p = cls(root, settings=settings)
        p.save()
        logger.info("Created project at %s", root)
        return p

    @classmethod
    def open(cls, root: Path) -> "Project":
        root = Path(root)
        if not (root / "project.json").exists():
            raise FileNotFoundError(f"No project.json at {root}")
        data = json.loads((root / "project.json").read_text())
        settings = AppSettings(**data.get("settings", {}))
        p = cls(root, settings=settings)
        for d in data.get("stimuli", []):
            p.stimuli[d["stimulus_id"]] = StimulusEntry.from_dict(d)
        return p

    def save(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        data = {
            "project_version": 1,
            "app_version": __version__,
            "settings": self.settings.model_dump(),
            "stimuli": [e.to_dict() for e in self.stimuli.values()],
        }
        (self.root / "project.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        return self.root / "project.json"

    # ---- Stimulus registration ----
    def _next_id(self, stem: str) -> str:
        base = "".join(c if c.isalnum() else "_" for c in stem)
        candidate = base
        i = 1
        while candidate in self.stimuli:
            candidate = f"{base}_{i}"
            i += 1
        return candidate

    def add_stimulus(self, file_path: str, media_type: MediaType, **meta) -> StimulusEntry:
        file_path = os.path.abspath(file_path)
        stem = Path(file_path).stem
        sid = self._next_id(stem)
        entry = StimulusEntry(
            stimulus_id=sid, file_path=file_path,
            file_name=Path(file_path).name, media_type=media_type.value,
            added_at=_now_iso(), **meta,
        )
        self.stimuli[sid] = entry
        self.save()
        return entry

    def add_folder(self, folder: str, media_type: MediaType, recursive: bool = False) -> List[StimulusEntry]:
        folder = Path(folder)
        pattern = "**/*" if recursive else "*"
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp",
                ".mp4", ".mov", ".avi", ".mkv", ".webm"}
        added: List[StimulusEntry] = []
        for p in sorted(folder.glob(pattern)):
            if not p.is_file() or p.suffix.lower() not in exts:
                continue
            if any(e.file_path == str(p) for e in self.stimuli.values()):
                continue
            entry = self.add_stimulus(str(p), media_type)
            added.append(entry)
        return added

    # ---- Result persistence ----
    def result_path(self, stimulus_id: str) -> Path:
        return self.results_dir / f"{stimulus_id}.json"

    def has_result(self, stimulus_id: str) -> bool:
        entry = self.stimuli.get(stimulus_id)
        return bool(entry) and self.result_path(stimulus_id).exists() and entry.status == "completed"

    def save_analysis(self, analysis: StimulusAnalysis) -> Path:
        path = self.result_path(analysis.stimulus_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
        entry = self.stimuli.get(analysis.stimulus_id)
        if entry:
            entry.status = "completed" if analysis.completed else "failed"
            entry.analysis_version = analysis.analysis_version
        self.save()
        return path

    def load_analysis(self, stimulus_id: str) -> Optional[StimulusAnalysis]:
        path = self.result_path(stimulus_id)
        if not path.exists():
            return None
        return StimulusAnalysis.model_validate_json(path.read_text(encoding="utf-8"))

    # ---- Mask persistence ----
    def mask_path(self, stimulus_id: str, person_id: str, kind: str = "final") -> Path:
        d = self.masks_dir / stimulus_id
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{person_id}__{kind}.png"

    def save_mask(self, stimulus_id: str, person_id: str, mask: np.ndarray, kind: str = "final") -> Path:
        path = self.mask_path(stimulus_id, person_id, kind)
        mask_u8 = (np.asarray(mask) > 0).astype(np.uint8) * 255
        cv2_write = __import__("cv2")
        cv2_write.imwrite(str(path), mask_u8)
        return path

    def load_mask(self, stimulus_id: str, person_id: str, kind: str = "final") -> Optional[np.ndarray]:
        import cv2
        path = self.mask_path(stimulus_id, person_id, kind)
        if not path.exists():
            return None
        return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    def reproducibility_report(self, stimulus_id: str) -> Optional[dict]:
        analysis = self.load_analysis(stimulus_id)
        if analysis is None:
            return None
        rep = analysis.reproducibility.model_dump()
        rep["app_version"] = __version__
        rep["stimulus_id"] = stimulus_id
        rep["completed"] = analysis.completed
        rep["record_version"] = analysis.analysis_version
        return rep


def _now_iso() -> str:
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
