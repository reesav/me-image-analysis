"""Export: flatten analyses into tidy tables (CSV, Excel) + data dictionary.

Tables are intentionally one-record-per-entity (no nested structures) so they
open cleanly in R, SPSS, Python, and Excel. Stable IDs join tables. Original
(automated) and reviewed values are distinguished via columns / the
correction history stored per record.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..storage.project import Project
from ..core.schemas import ReviewStatus

logger = logging.getLogger("artwork_analyzer.exports.tables")


def _load_analyses(project: Project) -> List:
    out = []
    for sid in project.stimuli:
        a = project.load_analysis(sid)
        if a is not None:
            out.append(a)
    return out


def _det_box(row, prefix):
    return {
        f"{prefix}_x": row.box.x,
        f"{prefix}_y": row.box.y,
        f"{prefix}_width": row.box.width,
        f"{prefix}_height": row.box.height,
        f"{prefix}_confidence": row.box.confidence,
    }


def stimulus_summary(project: Project) -> pd.DataFrame:
    rows = []
    for a in _load_analyses(project):
        m = a.image_metadata or {}
        c = a.color
        b = a.brightness_contrast
        comp = a.composition
        vm = a.video_metadata
        row = {
            "stimulus_id": a.stimulus_id,
            "file_name": a.file_name,
            "file_path": a.file_path,
            "media_type": a.media_type.value,
            "width": m.get("width"),
            "height": m.get("height"),
            "pixel_count": m.get("pixel_count"),
            "aspect_ratio": m.get("aspect_ratio"),
            "mode": m.get("mode"),
            "file_size_bytes": m.get("file_size_bytes"),
            "file_format": m.get("file_format"),
            "color_profile": m.get("color_profile"),
            "exif_orientation": m.get("exif_orientation"),
            "mean_luminance": (c.mean_luminance if c else None),
            "std_luminance": (b.std_luminance if b else None),
            "rms_contrast": (b.rms_contrast if b else None),
            "michelson_contrast": (b.michelson_contrast if b else None),
            "michelson_applicable": (b.michelson_applicable if b else None),
            "colorfulness": (c.colorfulness if c else None),
            "foreground_background_ratio": (comp.foreground_background_ratio if comp else None),
            "person_count": len(a.people),
            "face_count": len(a.faces),
            "object_count": len(a.objects),
            "scene_cut_count": len(a.scene_cuts),
            "motion_mean": (a.motion.mean_magnitude if a.motion else None),
            "video_duration_seconds": (vm.duration_seconds if vm else None),
            "video_frame_rate": (vm.frame_rate if vm else None),
            "completed": a.completed,
            "review_status": a.review_status.value,
            "processing_timestamp": a.reproducibility.processing_timestamp,
            "app_version": a.reproducibility.app_version,
            "person_detector_model": a.reproducibility.person_detector_model,
            "segmentation_backend": a.reproducibility.segmentation_backend,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def people_table(project: Project) -> pd.DataFrame:
    rows = []
    for a in _load_analyses(project):
        for p in a.people:
            r = {
                "stimulus_id": a.stimulus_id,
                "person_id": p.person_id,
                "detection_model": p.detection_model,
                "segmentation_model": p.segmentation_model,
                **_det_box(p.box, "box"),
                "box_area_percent": p.box_area_percent,
                "mask_area_pixels": p.mask_area_pixels,
                "mask_area_percent": p.mask_area_percent,
                "mask_available": p.mask_available,
                "mask_confidence": p.mask_confidence,
                "centroid_x": p.centroid_x,
                "centroid_y": p.centroid_y,
                "review_status": p.review_status.value,
                "review_note": p.review_note,
            }
            rows.append(r)
    return pd.DataFrame(rows)


def faces_table(project: Project) -> pd.DataFrame:
    rows = []
    for a in _load_analyses(project):
        for f in a.faces:
            r = {
                "stimulus_id": a.stimulus_id,
                "face_id": f.face_id,
                "detection_model": f.detection_model,
                **_det_box(f.box, "face"),
                "face_box_area_percent": f.box_area_percent,
                "is_cropped": f.is_cropped,
                "review_status": f.review_status.value,
                "review_note": f.review_note,
            }
            rows.append(r)
    return pd.DataFrame(rows)


def objects_table(project: Project) -> pd.DataFrame:
    rows = []
    for a in _load_analyses(project):
        for o in a.objects:
            r = {
                "stimulus_id": a.stimulus_id,
                "object_id": o.object_id,
                "label": o.label,
                "confidence": o.confidence,
                **_det_box(o.box, "box"),
                "review_status": o.review_status.value,
            }
            rows.append(r)
    return pd.DataFrame(rows)


def video_scenes_table(project: Project) -> pd.DataFrame:
    rows = []
    for a in _load_analyses(project):
        if not a.scene_cuts:
            continue
        n = len(a.scene_cuts)
        vm = a.video_metadata
        dur = vm.duration_seconds or 0.0
        cuts_per_min = (n / (dur / 60.0)) if dur > 0 else None
        ts = [c.timestamp_seconds for c in a.scene_cuts]
        gaps = [t - s for t, s in zip(ts[1:], ts)] if len(ts) > 1 else []
        rows.append({
            "stimulus_id": a.stimulus_id,
            "num_cuts": n,
            "cuts_per_minute": round(cuts_per_min, 4) if cuts_per_min is not None else None,
            "avg_scene_duration_seconds": (round(sum(gaps) / len(gaps), 4) if gaps else None),
            "min_scene_duration_seconds": (round(min(gaps), 4) if gaps else None),
            "max_scene_duration_seconds": (round(max(gaps), 4) if gaps else None),
        })
    return pd.DataFrame(rows)


def scene_cuts_table(project: Project) -> pd.DataFrame:
    rows = []
    for a in _load_analyses(project):
        for c in a.scene_cuts:
            rows.append({
                "stimulus_id": a.stimulus_id,
                "type": c.type,
                "timestamp_seconds": c.timestamp_seconds,
                "frame_index": c.frame_index,
            })
    return pd.DataFrame(rows)


def frame_samples_table(project: Project) -> pd.DataFrame:
    rows = []
    for a in _load_analyses(project):
        for fs in a.frame_samples:
            row = {
                "stimulus_id": a.stimulus_id,
                "frame_index": fs.frame_index,
                "timestamp_seconds": fs.timestamp_seconds,
                "person_count": fs.person_count,
                "face_count": fs.face_count,
                "person_area_percent": fs.person_area_percent,
            }
            if fs.brightness:
                row["rms_contrast"] = fs.brightness.rms_contrast
                row["mean_luminance"] = fs.brightness.mean_luminance
            rows.append(row)
    return pd.DataFrame(rows)


def processing_log_table(project: Project) -> pd.DataFrame:
    rows = []
    for a in _load_analyses(project):
        for w in a.warnings:
            rows.append({"stimulus_id": a.stimulus_id, "severity": "warning", "message": w})
        for e in a.errors:
            rows.append({"stimulus_id": a.stimulus_id, "severity": "error", "message": e})
    for log_path in project.logs_dir.glob("run_*.log"):
        try:
            for line in log_path.read_text(encoding="utf-8").splitlines():
                rows.append({"stimulus_id": "", "severity": "runlog", "message": line})
        except Exception:
            pass
    return pd.DataFrame(rows)


def metadata_table(project: Project) -> pd.DataFrame:
    return pd.DataFrame([e.to_dict() for e in project.stimuli.values()])


def export_all(project: Project, dest_dir: Optional[Path] = None,
               include_automated: bool = True, include_reviewed: bool = True) -> Path:
    """Export all tables to CSV + a single Excel workbook + data dictionary."""
    dest = Path(dest_dir) if dest_dir else Path(project.root / "exports")
    dest.mkdir(parents=True, exist_ok=True)

    tables: Dict[str, pd.DataFrame] = {
        "stimulus_summary": stimulus_summary(project),
        "people": people_table(project),
        "faces": faces_table(project),
        "objects": objects_table(project),
        "video_scenes": video_scenes_table(project),
        "scene_cuts": scene_cuts_table(project),
        "frame_samples": frame_samples_table(project),
        "processing_log": processing_log_table(project),
        "stimulus_metadata": metadata_table(project),
    }
    for name, df in tables.items():
        df.to_csv(dest / f"{name}.csv", index=False)

    xlsx_path = dest / "analysis_results.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    _write_data_dictionary(dest, tables, project)
    logger.info("Exported %d tables to %s", len(tables), dest)
    return dest


def _write_data_dictionary(dest: Path, tables: Dict[str, pd.DataFrame], project: Project) -> None:
    """Write a JSON + Markdown data dictionary describing fields, units, algorithms."""
    doc = {
        "generated_by": "artwork_analyzer",
        "description": "Data dictionary for exported tables.",
        "tables": {},
        "metric_definitions": {
            "relative_luminance": "sRGB relative luminance per ITU-R BT.709, range 0-1.",
            "rms_contrast": "Standard deviation of normalised luminance (Reinhard 2013), dimensionless.",
            "michelson_contrast": "(Lmax - Lmin)/(Lmax + Lmin); only when michelson_applicable=True (bimodal image).",
            "colorfulness": "Hasler-Susstrunk perceived colourfulness metric (dimensionless).",
            "person_mask_area_percent": "100 * foreground mask pixels / total image pixels (reviewed mask).",
            "box_area_percent": "100 * bounding-box area / total image pixels. Not actual body surface area.",
            "motion_magnitude": "Normalised dense optical-flow magnitude (0-1, Farneback); conflates camera and subject motion.",
        },
    }
    for name, df in tables.items():
        doc["tables"][name] = {
            "rows": int(len(df)),
            "fields": {c: f"dtype: {df[c].dtype}" for c in df.columns},
            "join_key": "stimulus_id" if "stimulus_id" in df.columns else None,
        }
    if project.stimuli:
        rep = project.reproducibility_report(list(project.stimuli)[0])
        if rep:
            doc["sample_reproducibility"] = rep
    (dest / "data_dictionary.json").write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    lines = ["# Data Dictionary", "", "## Metric definitions", ""]
    for k, v in doc["metric_definitions"].items():
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## Tables", ""]
    for name, df in tables.items():
        lines.append(f"### {name} ({len(df)} rows)")
        for c in df.columns:
            lines.append(f"- `{c}`: {df[c].dtype}")
        lines.append("")
    (dest / "data_dictionary.md").write_text("\n".join(lines), encoding="utf-8")
