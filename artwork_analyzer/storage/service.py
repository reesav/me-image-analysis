"""Batch processing service: orchestrate analysis and persistence.

Runs analysis per stimulus, persists results, and records a processing log
per run. Supports skipping already-completed analyses (resume) and reports
per-file progress and errors. Processing is sequential with a progress hook so
the UI can render a progress bar; each result is persisted immediately so an
interrupted batch can be resumed.
"""

from __future__ import annotations

import io
import logging
import time
import traceback
from contextlib import redirect_stdout
from typing import Callable, Iterable, List, Optional

import numpy as np

from .project import Project
from ..core.config import AppSettings
from ..core.schemas import StimulusAnalysis, MediaType
from ..analysis.analyzer import analyze_image
from ..video.analyzer import analyze_video

logger = logging.getLogger("artwork_analyzer.storage.service")

ProgressCallback = Callable[[int, int, str], None]


def _detect_media_type(file_path: str) -> MediaType:
    ext = file_path.lower().rsplit(".", 1)[-1] if "." in file_path else ""
    if ext in ("mp4", "mov", "avi", "mkv", "webm", "flv", "wmv"):
        return MediaType.VIDEO
    return MediaType.IMAGE


def _is_image(p: str) -> bool:
    ext = p.lower().rsplit(".", 1)[-1] if "." in p else ""
    return ext in {"jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"}


def _is_video(p: str) -> bool:
    return not _is_image(p)


class ProcessingService:
    def __init__(self, project: Project, settings: Optional[AppSettings] = None):
        self.project = project
        self.settings = settings or project.settings
        self.last_run_log: List[str] = []

    def process_one(self, file_path: str, stimulus_id: Optional[str] = None) -> StimulusAnalysis:
        settings = self.settings
        if _is_video(file_path):
            analysis = analyze_video(file_path, settings, stimulus_id=stimulus_id)
        else:
            analysis = analyze_image(file_path, settings, stimulus_id=stimulus_id)
            self._persist_automated_masks(file_path, analysis)
        self.project.save_analysis(analysis)
        return analysis
        return analysis

    def _persist_automated_masks(self, file_path: str, analysis: StimulusAnalysis) -> None:
        """Re-derive and store the automated person masks as PNG (for review)."""
        if not analysis.people:
            return
        try:
            from ..analysis.image_io import load_image
            from ..analysis.segmentation import segment_people
            img = load_image(file_path, stimulus_id=analysis.stimulus_id)
            masks = segment_people(img.display_array, analysis.people, self.settings.detector)
            # Reset people list to the fresh records (keeps area metrics from analyzer).
            for pid, mask, conf, backend in masks:
                self.project.save_mask(analysis.stimulus_id, pid, mask, kind="automated")
        except Exception as exc:
            logger.warning("Could not persist automated masks for %s: %s",
                           analysis.stimulus_id, exc)

    def process_batch(
        self,
        stimulus_ids: Optional[Iterable[str]] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> dict:
        """Process stimuli. Returns a summary dict. Skips completed if configured."""
        ids = list(stimulus_ids) if stimulus_ids is not None else list(self.project.stimuli.keys())
        total = len(ids)
        completed = 0
        failed = 0
        skipped = 0
        log_buf: List[str] = []

        for i, sid in enumerate(ids, 1):
            entry = self.project.stimuli[sid]
            if self.settings.processing.skip_completed and self.project.has_result(sid):
                skipped += 1
                log_buf.append(f"[skip] {sid}: already completed")
                if progress:
                    progress(i, total, f"skipped {sid}")
                continue

            fp = entry.file_path
            if not __import__("os").path.exists(fp):
                failed += 1
                log_buf.append(f"[fail] {sid}: file missing -> {fp}")
                self.project.stimuli[sid].status = "failed"
                continue
            try:
                before = time.time()
                buf = io.StringIO()
                # Capture log lines for this run.
                handler = logging.StreamHandler(buf)
                logger.addHandler(handler)
                analysis = self.process_one(fp, stimulus_id=sid)
                logger.removeHandler(handler)
                elapsed = time.time() - before
                status = "completed" if analysis.completed else "failed"
                completed += 1 if analysis.completed else 0
                failed += 0 if analysis.completed else 1
                msg = f"[{status}] {sid} ({entry.media_type}) in {elapsed:.1f}s "
                msg += f"people={len(analysis.people)} faces={len(analysis.faces)}"
                log_buf.append(msg)
                if progress:
                    progress(i, total, msg)
            except Exception as exc:
                failed += 1
                tb = traceback.format_exc(limit=2)
                log_buf.append(f"[fail] {sid}: {exc}\n{tb}")
                self.project.stimuli[sid].status = "failed"
                if progress:
                    progress(i, total, f"failed {sid}: {exc}")
                logger.exception("Failed processing %s", sid)

        self.last_run_log = log_buf
        log_path = self.project.logs_dir / f"run_{int(time.time())}.log"
        log_path.write_text("\n".join(log_buf), encoding="utf-8")
        return {
            "total": total, "completed": completed, "failed": failed,
            "skipped": skipped, "log_path": str(log_path),
        }
