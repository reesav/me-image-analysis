"""Streamlit UI entry point.

Run with:  streamlit run artwork_analyzer/ui/app.py

Provides a research dashboard with navigation:
  Dashboard | Import | Analyze | Review | Results | Batch | Settings
The active Project is kept in st.session_state across reruns.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

# Bootstrap the project root onto sys.path so `artwork_analyzer.*` imports resolve
# regardless of how Streamlit launches this file (script path may be ui/).
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import streamlit as st

from artwork_analyzer.core.config import AppSettings, save_settings, default_config_path
from artwork_analyzer.core.version import __version__
from artwork_analyzer.storage.project import Project
from artwork_analyzer.storage.service import ProcessingService
from artwork_analyzer.analysis.analyzer import analyze_image
from artwork_analyzer.analysis.image_io import load_image
from artwork_analyzer.analysis.segmentation import segment_people, apply_masks
from artwork_analyzer.exports.tables import export_all, stimulus_summary
from artwork_analyzer.core.schemas import MediaType, ReviewStatus, StimulusAnalysis
from artwork_analyzer.ui.components import render_overlay, pil_image

st.set_page_config(page_title="Artwork Analyzer", layout="wide")


def _project() -> Optional[Project]:
    return st.session_state.get("project")


def _settings() -> AppSettings:
    return st.session_state.get("settings", AppSettings())


def _ensure_settings():
    if "settings" not in st.session_state:
        try:
            st.session_state.settings = AppSettings()
            save_settings(st.session_state.settings)
        except Exception:
            st.session_state.settings = AppSettings()


def page_dashboard():
    st.title("Artwork Analyzer — Dashboard")
    _ensure_settings()
    proj = _project()
    st.caption(f"Application version {__version__}")
    if proj is None:
        st.info("Open or create a project from the sidebar.")
        return
    entries = list(proj.stimuli.values())
    n_total = len(entries)
    n_completed = sum(1 for e in entries if e.status == "completed")
    n_failed = sum(1 for e in entries if e.status == "failed")
    images = sum(1 for e in entries if e.media_type == "image")
    videos = sum(1 for e in entries if e.media_type == "video")
    cols = st.columns(6)
    cols[0].metric("Stimuli", n_total)
    cols[1].metric("Completed", n_completed)
    cols[2].metric("Failed", n_failed)
    cols[3].metric("Images", images)
    cols[4].metric("Videos", videos)
    cols[5].metric("Awaiting review", sum(1 for e in entries if e.status != "completed"))
    if entries:
        df = stimulus_summary(proj)
        st.dataframe(df, use_container_width=True)


def page_import():
    st.title("Import Stimuli")
    proj = _project()
    if proj is None:
        st.warning("Create a project first.")
        return
    folder = st.text_input("Folder to import", value="")
    recursive = st.checkbox("Recursive", value=False)
    media = st.radio("Media type", ["image", "video", "auto"], horizontal=True)
    if st.button("Scan folder") and folder:
        p = Path(folder)
        if not p.is_dir():
            st.error("Folder does not exist")
            return
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp",
                ".mp4", ".mov", ".avi", ".mkv", ".webm"}
        files = [x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in exts] if recursive else [x for x in p.iterdir() if x.is_file() and x.suffix.lower() in exts]
        st.write(f"Found {len(files)} files")
        for f in files[:5]:
            st.caption(f.name)
        if st.button("Import found files"):
            mt = MediaType.IMAGE if media == "image" else (MediaType.VIDEO if media == "video" else None)
            if mt is None:
                exts_img = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
                added = 0
                for f in files:
                    _mt = MediaType.VIDEO if f.suffix.lower() not in exts_img else MediaType.IMAGE
                    proj.add_stimulus(str(f), _mt)
                    added += 1
            else:
                for f in files:
                    proj.add_stimulus(str(f), mt)
            proj.save()
            st.success(f"Imported stimuli (total: {len(proj.stimuli)})")
            st.rerun()


def page_analyze():
    st.title("Analyze a Stimulus")
    proj = _project()
    if proj is None or not proj.stimuli:
        st.info("Import stimuli first.")
        return
    sid = st.selectbox("Stimulus", list(proj.stimuli.keys()), format_func=lambda s: proj.stimuli[s].file_name)
    entry = proj.stimuli[sid]
    if not os.path.exists(entry.file_path):
        st.error("Source file is missing: " + entry.file_path)
        return
    if st.button("Run analysis"):
        with st.spinner("Analyzing..."):
            svc = ProcessingService(proj, _settings())
            res = svc.process_one(entry.file_path, stimulus_id=sid)
        st.success("Analysis complete")
        st.session_state[f"_last_{sid}"] = res
        st.rerun()
    existing = proj.load_analysis(sid)
    if existing:
        st.subheader("Results overview")
        _render_results_table(existing)
        _render_workspace(proj, existing)
    last = st.session_state.get(f"_last_{sid}")
    if last:
        _render_workspace(proj, last)


def _render_results_table(res):
    st.caption(f"Completed: {res.completed}")
    if res.errors:
        st.error("Errors: " + "; ".join(res.errors))
    if res.warnings:
        st.warning("Warnings: " + "; ".join(res.warnings))
    if res.image_metadata:
        st.json(res.image_metadata)
    rep = res.reproducibility
    st.caption(f"Models: person={rep.person_detector_model}, face={rep.face_detector_model}, "
               f"seg={rep.segmentation_backend}, device={rep.device}, ts={rep.processing_timestamp}")


def _render_workspace(proj: Project, res):
    st.subheader("Workspace")
    if not os.path.exists(res.file_path):
        st.error("Cannot render: file missing")
        return
    img = load_image(res.file_path, stimulus_id=res.stimulus_id)
    disp = img.display_array
    # Load automated masks if present.
    masks = {}
    for p in res.people:
        m = proj.load_mask(res.stimulus_id, p.person_id, "final") \
            or proj.load_mask(res.stimulus_id, p.person_id, "automated")
        if m is not None:
            masks[p.person_id] = m
    overlay = render_overlay(disp, people=res.people, faces=res.faces, masks=masks)
    st.image(pil_image(overlay), caption=f"{res.file_name} (reviewed)" if res.review_status.value == "reviewed_accepted" else f"{res.file_name} (automated)", use_column_width=True)
    # Measurement cards
    c = res.color
    if c:
        cols = st.columns(4)
        cols[0].metric("Mean luminance", f"{c.mean_luminance:.3f}")
        cols[1].metric("RMS contrast", f"{res.brightness_contrast.rms_contrast:.3f}" if res.brightness_contrast else "—")
        cols[2].metric("Colorfulness", f"{c.colorfulness:.2f}")
        cols[3].metric("Foreground ratio", f"{res.composition.foreground_background_ratio:.3f}" if res.composition else "—")
    # People table
    if res.people:
        st.markdown("**Detected people**")
        df = pd.DataFrame([{
            "person_id": p.person_id, "confidence": round(p.box.confidence, 3),
            "box_area_%": round(p.box_area_percent, 2),
            "mask_area_%": (round(p.mask_area_percent, 2) if p.mask_area_percent is not None else None),
            "mask_available": p.mask_available,
            "review_status": p.review_status.value,
        } for p in res.people])
        st.dataframe(df, use_container_width=True)
    # Faces table
    if res.faces:
        st.markdown("**Detected faces**")
        fdf = pd.DataFrame([{
            "face_id": f.face_id, "confidence": round(f.box.confidence, 3),
            "box_area_%": round(f.box_area_percent, 2),
            "cropped": f.is_cropped,
        } for f in res.faces])
        st.dataframe(fdf, use_container_width=True)


def page_review():
    st.title("Review Detections")
    proj = _project()
    if proj is None:
        return
    sid = st.selectbox("Stimulus", [s for s in proj.stimuli if proj.has_result(s)],
                       format_func=lambda s: proj.stimuli[s].file_name)
    if not sid:
        st.info("No completed analyses to review.")
        return
    res = proj.load_analysis(sid)
    if res is None:
        st.info("No analysis result for this stimulus.")
        return
    # Re-segment / reload masks for interactive review.
    entry = proj.stimuli[sid]
    display = None
    mask_map = {}
    if os.path.exists(entry.file_path):
        img = load_image(entry.file_path, stimulus_id=sid)
        display = img.display_array
        # Prefer previously-finalised masks; fall back to fresh GrabCut.
        for p in res.people:
            prev = proj.load_mask(sid, p.person_id, "final")
            if prev is not None:
                mask_map[p.person_id] = prev
        if res.people:
            masks = segment_people(display, res.people, _settings().detector)
            apply_masks(res.people, masks)
            for pid, m, _, _ in masks:
                if pid not in mask_map:
                    mask_map[pid] = m
    else:
        st.error("Source file is missing: " + entry.file_path)
        return

    overlay = render_overlay(display, people=res.people, faces=res.faces, masks=mask_map)
    st.image(pil_image(overlay), caption="Review overlay (boxes=person, lines=face, translucent=mask)", use_column_width=True)

    st.markdown("**Per-person review**")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Accept all"):
            for p in res.people:
                p.review_status = ReviewStatus.REVIEWED_ACCEPTED
    with col_b:
        if st.button("Reject all"):
            for p in res.people:
                p.review_status = ReviewStatus.REJECTED
            st.rerun()

    statuses = [r.value for r in ReviewStatus]
    for p in res.people:
        c1, c2 = st.columns([2, 3])
        with c1:
            st.write(p.person_id)
        with c2:
            default = statuses.index(p.review_status.value)
            chosen = st.radio("status", statuses, index=default,
                              key=f"st_{sid}_{p.person_id}", horizontal=True)
            if chosen != p.review_status.value:
                p.review_status = ReviewStatus(chosen)

    if st.button("Save review"):
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for p in res.people:
            if p.review_status in (ReviewStatus.REVIEWED_ACCEPTED, ReviewStatus.REVIEWED_MODIFIED):
                mask = mask_map.get(p.person_id)
                if mask is not None:
                    proj.save_mask(sid, p.person_id, mask, kind="final")
            if p.review_status != ReviewStatus.AUTOMATED:
                res.correction_history.append({
                    "timestamp": ts,
                    "person_id": p.person_id,
                    "action": p.review_status.value,
                    "reviewer": "ui",
                })
        if res.people and all(p.review_status == ReviewStatus.REVIEWED_ACCEPTED for p in res.people):
            res.review_status = ReviewStatus.REVIEWED_ACCEPTED
        elif any(p.review_status != ReviewStatus.AUTOMATED for p in res.people):
            res.review_status = ReviewStatus.REVIEWED_MODIFIED
        proj.save_analysis(res)
        entry = proj.stimuli[sid]
        entry.status = "reviewed"
        proj.save()
        st.success("Review persisted: final masks + correction_history saved")
        st.rerun()


def page_quick_check():
    st.title("Quick Check — Upload & Analyze")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"])
    if not uploaded:
        st.info("Drop an image here to analyze it.")
        return

    import tempfile
    from pathlib import Path as _P
    from artwork_analyzer.analysis.analyzer import analyze_image

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded.name.rsplit(".", 1)[-1])
    tmp.write(uploaded.getvalue())
    tmp.close()

    with st.spinner("Analyzing image..."):
        res = analyze_image(tmp.name, _settings(), stimulus_id=uploaded.name)

    if not res.completed:
        st.error("Analysis failed: " + "; ".join(res.errors))
        return

    st.subheader("Result: " + uploaded.name)
    c = res.color
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean luminance", f"{c.mean_luminance:.3f}" if c else "—")
    col2.metric("RMS contrast", f"{res.brightness_contrast.rms_contrast:.3f}" if res.brightness_contrast else "—")
    col3.metric("Colorfulness", f"{c.colorfulness:.2f}" if c and c.colorfulness else "—")
    col4.metric("Foreground ratio", f"{res.composition.foreground_background_ratio:.3f}" if res.composition else "—")

    # Render overlay
    from artwork_analyzer.analysis.image_io import load_image
    img = load_image(res.file_path, stimulus_id=res.stimulus_id)
    overlay = render_overlay(img.display_array, people=res.people, faces=res.faces)
    st.image(pil_image(overlay), caption="Overlay (person boxes + face boxes)", use_column_width=True)

    # Clean up temp upload
    try:
        os.unlink(tmp.name)
    except OSError:
        pass

    st.markdown(f"**People detected:** {len(res.people)}   |   **Faces detected:** {len(res.faces)}   |   **Objects:** {len(res.objects)}")
    if res.people:
        import pandas as pd
        df = pd.DataFrame([{
            "person_id": p.person_id, "confidence": round(p.box.confidence, 3),
            "box_area_%": round(p.box_area_percent, 2),
            "mask_area_%": (round(p.mask_area_percent, 2) if p.mask_area_percent is not None else None),
            "review_status": p.review_status.value,
        } for p in res.people])
        st.dataframe(df, use_container_width=True)
    if res.faces:
        import pandas as pd
        fdf = pd.DataFrame([{
            "face_id": f.face_id, "confidence": round(f.box.confidence, 3),
            "box_area_%": round(f.box_area_percent, 2),
            "cropped": f.is_cropped,
        } for f in res.faces])
        st.dataframe(fdf, use_container_width=True)
    if res.errors:
        st.error("Errors: " + "; ".join(res.errors))
    if res.warnings:
        st.warning("Warnings: " + "; ".join(res.warnings))


def page_results():
    st.title("Results Explorer")
    proj = _project()
    if proj is None:
        return
    df = stimulus_summary(proj)
    if df.empty:
        st.info("No results yet.")
        return
    st.dataframe(df, use_container_width=True)
    if st.button("Export all tables (CSV + Excel)"):
        dest = export_all(proj)
        st.success(f"Exported to {dest}")


def page_batch():
    st.title("Batch Processing")
    proj = _project()
    if proj is None or not proj.stimuli:
        st.info("Import stimuli first.")
        return
    if st.button("Run batch analysis"):
        svc = ProcessingService(proj, _settings())
        bar = st.progress(0)
        status = st.empty()
        results = {"completed": 0, "failed": 0, "skipped": 0}

        def cb(i, total, msg):
            bar.progress(i / total)
            status.text(f"{i}/{total}: {msg}")
        summary = svc.process_batch(progress=cb)
        st.success(f"Batch done: {summary}")
        st.text("\n".join(svc.last_run_log))


def page_settings():
    st.title("Settings")
    settings = _settings()
    st.json(settings.model_dump())
    d = st.detector if False else None
    s = st.session_state
    st.subheader("Detection")
    s["person_detector"] = st.text_input("Person detector", value=settings.detector.person_detector)
    s["face_detector"] = st.text_input("Face detector", value=settings.detector.face_detector)
    s["segmentation_backend"] = st.text_input("Segmentation backend", value=settings.detector.segmentation_backend)
    st.subheader("Confidence thresholds")
    settings.detector.person_confidence_threshold = st.slider("Person confidence", 0.0, 1.0, settings.detector.person_confidence_threshold)
    settings.detector.face_confidence_threshold = st.slider("Face confidence", 0.0, 1.0, settings.detector.face_confidence_threshold)
    settings.detector.face_score_threshold = st.slider("Face score (OpenCV internal)", 0.0, 1.0, settings.detector.face_score_threshold)
    settings.detector.face_nms_threshold = st.slider("Face NMS (overlap suppression)", 0.0, 1.0, settings.detector.face_nms_threshold, help="Lower = keep faces that are close together")
    st.subheader("Video")
    settings.video.frame_interval_seconds = st.number_input("Frame interval (s)", 0.1, 60.0, settings.video.frame_interval_seconds)
    settings.video.scene_cut_threshold = st.slider("Scene-cut threshold (0-255)", 0.0, 255.0, settings.video.scene_cut_threshold)
    st.subheader("Device")
    settings.detector.device = st.text_input("Device (cpu/cuda)", value=settings.detector.device)
    if st.button("Save settings"):
        save_settings(settings)
        st.session_state.settings = settings
        st.success("Settings saved")


PROJECTS_KEY = "recent_projects"


def _open_project():
    recent = st.session_state.get(PROJECTS_KEY, [])
    default_root = str(Path.home() / "artwork_projects" / "default")
    root = st.sidebar.text_input("Project folder", value=default_root)
    if st.sidebar.button("Open / create project"):
        from pathlib import Path as _P
        p = _P(root)
        if (p / "project.json").exists():
            proj = Project.open(p)
        else:
            proj = Project.create(p, settings=_settings())
        st.session_state["project"] = proj
        recent = [r for r in (recent + [str(p)]) if r != str(p)] + [str(p)]
        st.session_state[PROJECTS_KEY] = recent[-10:]
        st.rerun()


def main():
    _ensure_settings()
    _open_project()
    st.sidebar.title("Artwork Analyzer")
    if _project() is None:
        st.sidebar.info("No project opened — use Quick Check below (no setup needed).")
        page_quick_check()
        st.sidebar.divider()
        page_settings()
        return
    pages = {
        "Quick Check": page_quick_check,
        "Dashboard": page_dashboard,
        "Import": page_import,
        "Analyze": page_analyze,
        "Review": page_review,
        "Results": page_results,
        "Batch": page_batch,
        "Settings": page_settings,
    }
    page = st.sidebar.radio("Navigate", list(pages.keys()))
    pages[page]()


if __name__ == "__main__":
    main()
