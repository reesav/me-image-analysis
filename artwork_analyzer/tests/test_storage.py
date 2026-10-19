"""Tests for project storage, result persistence, and exports."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from artwork_analyzer.core.config import AppSettings, DetectorSettings
from artwork_analyzer.core.schemas import MediaType
from artwork_analyzer.storage.project import Project
from artwork_analyzer.storage.service import ProcessingService
from artwork_analyzer.exports.tables import export_all
from artwork_analyzer.analysis.segmentation import segment_people, apply_masks
from artwork_analyzer.core.schemas import PersonRecord, DetectionBox, ReviewStatus


def _make_image(path, arr):
    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def test_project_create_save_load(tmp_path, two_rectangles):
    proj = Project.create(tmp_path / "proj")
    entry = proj.add_stimulus("/tmp/unused.png", MediaType.IMAGE, group="g1", condition="c1")
    # Persist a fixture image.
    img_path = tmp_path / "stim.png"
    _make_image(img_path, two_rectangles)
    proj.stimuli[entry.stimulus_id].file_path = str(img_path)
    proj.save()

    # Reopen and verify persistence.
    proj2 = Project.open(tmp_path / "proj")
    assert proj2.stimuli[entry.stimulus_id].group == "g1"
    assert proj2.has_result(entry.stimulus_id) is False


def test_process_and_persist(tmp_path, two_rectangles):
    img_path = tmp_path / "stim.png"
    _make_image(img_path, two_rectangles)
    proj = Project.create(tmp_path / "proj")
    proj.add_stimulus(str(img_path), MediaType.IMAGE)
    settings = AppSettings(detector=DetectorSettings(person_detector="disabled"))
    svc = ProcessingService(proj, settings)
    res = svc.process_one(str(img_path), stimulus_id=list(proj.stimuli.keys())[0])
    assert res.completed
    assert proj.has_result(list(proj.stimuli.keys())[0])

    # Reload from disk.
    loaded = proj.load_analysis(list(proj.stimuli.keys())[0])
    assert loaded is not None
    assert loaded.color.mean_luminance == pytest.approx(res.color.mean_luminance)
    rep = proj.reproducibility_report(list(proj.stimuli.keys())[0])
    assert rep["app_version"] == "1.0.0"


def test_mask_persistence(tmp_path, two_rectangles):
    proj = Project.create(tmp_path / "proj")
    img_path = tmp_path / "stim.png"
    _make_image(img_path, two_rectangles)
    entry = proj.add_stimulus(str(img_path), MediaType.IMAGE)
    p = PersonRecord(
        person_id="person_000",
        box=DetectionBox(x=0.1, y=0.2, width=0.2, height=0.7, confidence=0.9, label="person"),
        box_area_percent=0.0,
    )
    display = cv2.cvtColor(two_rectangles, cv2.COLOR_RGB2BGR)
    display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    masks = segment_people(display, [p], AppSettings().detector)
    apply_masks([p], masks)
    saved = proj.save_mask(entry.stimulus_id, p.person_id, masks[0][1], kind="automated")
    assert saved.exists()
    loaded = proj.load_mask(entry.stimulus_id, p.person_id, "automated")
    assert loaded is not None
    assert int((loaded > 0).sum()) > 0


def test_export_produces_files(tmp_path, two_rectangles):
    img_path = tmp_path / "stim.png"
    _make_image(img_path, two_rectangles)
    proj = Project.create(tmp_path / "proj")
    sid = proj.add_stimulus(str(img_path), MediaType.IMAGE).stimulus_id
    svc = ProcessingService(proj, AppSettings(detector=DetectorSettings(person_detector="disabled")))
    svc.process_one(str(img_path), stimulus_id=sid)
    dest = export_all(proj, dest_dir=tmp_path / "out")
    assert (dest / "stimulus_summary.csv").exists()
    assert (dest / "people.csv").exists()
    assert (dest / "faces.csv").exists()
    assert (dest / "data_dictionary.md").exists()
    assert (dest / "data_dictionary.json").exists()
    xlsx = dest / "analysis_results.xlsx"
    assert xlsx.exists() and xlsx.stat().st_size > 0
    import pandas as pd
    xl = pd.read_excel(xlsx, sheet_name="stimulus_metadata")
    assert len(xl) == 1


def test_review_persistence_wiring(tmp_path, two_rectangles):
    """Exercises the Review page's end-to-end persistence path."""
    img_path = tmp_path / "stim.png"
    _make_image(img_path, two_rectangles)
    proj = Project.create(tmp_path / "proj")
    sid = proj.add_stimulus(str(img_path), MediaType.IMAGE).stimulus_id
    svc = ProcessingService(proj, AppSettings(detector=DetectorSettings(person_detector="disabled")))
    svc.process_one(str(img_path), stimulus_id=sid)

    res = proj.load_analysis(sid)
    assert res is not None
    # Simulate a manually-placed person that GrabCut can refine into a mask.
    p = PersonRecord(
        person_id="person_000",
        box=DetectionBox(x=0.1, y=0.2, width=0.2, height=0.6, confidence=0.9, label="person"),
        box_area_percent=0.0,
    )
    res.people.append(p)

    display = cv2.cvtColor(two_rectangles, cv2.COLOR_RGB2BGR)
    display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    masks = segment_people(display, [p], AppSettings().detector)
    apply_masks([p], masks)
    final_mask = masks[0][1]

    # Accept + persist final mask (mirrors page_review "Save review").
    p.review_status = ReviewStatus.REVIEWED_ACCEPTED
    proj.save_mask(sid, p.person_id, final_mask, kind="final")
    ts = "2026-10-15T00:00:00Z"
    res.correction_history.append({"timestamp": ts, "person_id": p.person_id,
                                   "action": p.review_status.value, "reviewer": "test"})
    res.review_status = ReviewStatus.REVIEWED_ACCEPTED
    proj.save_analysis(res)
    proj.stimuli[sid].status = "reviewed"
    proj.save()

    # Reload a fresh Project to confirm everything round-trips to disk.
    again = Project.open(tmp_path / "proj")
    r2 = again.load_analysis(sid)
    assert r2 is not None
    assert r2.people[0].review_status == ReviewStatus.REVIEWED_ACCEPTED
    assert r2.correction_history and r2.correction_history[0]["action"] == "reviewed_accepted"
    assert again.stimuli[sid].status == "reviewed"
    reloaded_mask = again.load_mask(sid, p.person_id, "final")
    assert reloaded_mask is not None
    assert int((reloaded_mask > 0).sum()) == int((final_mask > 0).sum())
