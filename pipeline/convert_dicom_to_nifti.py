# pipeline/convert_dicom_to_nifti.py
from __future__ import annotations

"""DICOM -> NIfTI converter for Radiomics10 (CT + PET) with robust series selection.

This rewrite fixes the specific failure mode you saw in multiple patients:
- Selecting a small-coverage thorax CT while selecting a whole-body PET
- Or selecting CT/PET from different studies/frames

Both lead to a formally "successful" registration that is visually awful (PET resampled into
an incompatible CT space / outside the body FOV).

Policy in this script
1) Pick PET first (best PET).
2) Require CT from the SAME StudyInstanceUID as the PET (hard requirement), and prefer
   matching FrameOfReferenceUID.
3) If no CT exists in that PET study: fallback to FrameOfReferenceUID match.
4) If still no match: do CT-only (do NOT generate pet_in_ctspace outputs).
5) Guard against WB PET + tiny CT span mismatch. If mismatch detected and no better CT
   exists, skip pet_in_ctspace outputs and mark status as mismatch.
6) No silent trimming: if cleaning drops too many slices, reject that series and try another.

The rest (SUVbw conversion, registration, QC, atomic writes, index CSV) stays consistent
with the project expectations.
"""

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from tqdm import tqdm

from pipeline.config import Config, ensure_project_dirs, verify_write_access


# ----------------------------
# Small utilities
# ----------------------------

def _load_index(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _merge_index(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        return new_rows
    if new_rows is None or new_rows.empty:
        return existing

    if "patient_id" not in existing.columns or "patient_id" not in new_rows.columns:
        return pd.concat([existing, new_rows], ignore_index=True)

    existing = existing.copy().drop_duplicates(subset=["patient_id"], keep="last").set_index("patient_id")
    new_rows = new_rows.copy().drop_duplicates(subset=["patient_id"], keep="last").set_index("patient_id")

    merged = existing.combine_first(new_rows)
    merged.update(new_rows)
    return merged.reset_index()


def _save_manifest(case_dir: Path, manifest: dict) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def _tmp_with_same_suffixes(out_path: Path) -> Path:
    suffixes = "".join(out_path.suffixes)
    base = out_path.name[:-len(suffixes)] if suffixes else out_path.name
    return out_path.with_name(f"{base}.tmp{suffixes}")


def _write_nifti(img: sitk.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_with_same_suffixes(out_path)
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass
    sitk.WriteImage(img, str(tmp), useCompression=True)
    os.replace(str(tmp), str(out_path))


def _safe_lower(s: Any) -> str:
    return str(s or "").strip().lower()


def _as_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _as_str_list(v) -> List[str]:
    if v is None:
        return []
    try:
        if isinstance(v, (list, tuple)):
            items = [str(x).strip() for x in v]
        else:
            s = str(v).strip()
            if not s:
                return []
            items = [p.strip() for p in s.split("\\")]
        return [x.upper() for x in items if x]
    except Exception:
        return []


# ----------------------------
# DICOM scanning + metadata
# ----------------------------

@dataclass(frozen=True)
class SeriesInfo:
    series_uid: str
    modality: str
    series_desc: str
    study_uid: str
    frame_uid: str
    num_files: int
    first_file: Path
    slice_thickness: Optional[float]
    pixel_spacing: Optional[Tuple[float, float]]
    image_type: str
    samples_per_pixel: int
    photometric: str


@dataclass
class SeriesCacheEntry:
    z_stats: dict
    geom_flags: List[str]
    cleaned_files: List[Path]
    clean_meta: dict


def _read_header(dcm_path: Path) -> dict:
    d = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)

    def s(tag: str) -> str:
        try:
            return str(getattr(d, tag, "") or "").strip()
        except Exception:
            return ""

    modality = s("Modality")
    series_uid = s("SeriesInstanceUID")
    study_uid = s("StudyInstanceUID")
    frame_uid = s("FrameOfReferenceUID")
    series_desc = s("SeriesDescription")

    image_type = ""
    try:
        it = getattr(d, "ImageType", "")
        if isinstance(it, (list, tuple)):
            image_type = "\\".join([str(x) for x in it])
        else:
            image_type = str(it)
    except Exception:
        image_type = ""

    slice_thickness = None
    try:
        slice_thickness = float(getattr(d, "SliceThickness", None))
    except Exception:
        slice_thickness = None

    pixel_spacing = None
    try:
        ps = getattr(d, "PixelSpacing", None)
        if ps is not None and len(ps) >= 2:
            pixel_spacing = (float(ps[0]), float(ps[1]))
    except Exception:
        pixel_spacing = None

    samples_per_pixel = 1
    try:
        samples_per_pixel = int(getattr(d, "SamplesPerPixel", 1) or 1)
    except Exception:
        samples_per_pixel = 1

    photometric = ""
    try:
        photometric = str(getattr(d, "PhotometricInterpretation", "") or "")
    except Exception:
        photometric = ""

    return {
        "modality": modality,
        "series_uid": series_uid,
        "study_uid": study_uid,
        "frame_uid": frame_uid,
        "series_desc": series_desc,
        "image_type": image_type,
        "slice_thickness": slice_thickness,
        "pixel_spacing": pixel_spacing,
        "samples_per_pixel": samples_per_pixel,
        "photometric": photometric,
    }


def _is_scalar_series(info: SeriesInfo) -> bool:
    spp = int(info.samples_per_pixel) if info.samples_per_pixel else 1
    photo = (info.photometric or "").strip().upper()
    if spp != 1:
        return False
    if photo in ("RGB", "YBR_FULL", "YBR_FULL_422", "PALETTE COLOR"):
        return False
    return True


def _is_localizer(info: SeriesInfo, localizer_keywords: Tuple[str, ...]) -> bool:
    sd = _safe_lower(info.series_desc)
    it = _safe_lower(info.image_type)
    return any(k in sd for k in localizer_keywords) or any(k in it for k in localizer_keywords)


def _z_from_dicom(dcm_path: Path) -> Optional[float]:
    try:
        d = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
        ipp = getattr(d, "ImagePositionPatient", None)
        if ipp is None or len(ipp) < 3:
            return None
        return float(ipp[2])
    except Exception:
        return None


def _series_z_stats(series_files: List[Path]) -> dict:
    zs: List[float] = []
    for p in series_files:
        z = _z_from_dicom(p)
        if z is not None and np.isfinite(z):
            zs.append(z)

    if len(zs) < 5:
        return {"n": len(zs), "span_mm": None, "dz_med": None, "dz_max": None, "ratio": None}

    z = np.sort(np.array(zs, dtype=np.float64))
    dz = np.abs(np.diff(z))
    dz = dz[dz > 1e-6]
    span = float(z[-1] - z[0])

    if dz.size == 0:
        return {"n": len(zs), "span_mm": span, "dz_med": None, "dz_max": None, "ratio": None}

    dz_med = float(np.median(dz))
    dz_max = float(np.max(dz))
    ratio = float(dz_max / dz_med) if dz_med > 0 else float("inf")
    return {"n": len(zs), "span_mm": span, "dz_med": dz_med, "dz_max": dz_max, "ratio": ratio}


def _geometry_flags(zs: dict, slice_thickness: Optional[float], modality: str) -> List[str]:
    flags: List[str] = []
    if zs.get("span_mm") is None or zs.get("dz_med") is None:
        flags.append("no_z_stats")
        return flags

    dz_med = float(zs["dz_med"])
    dz_max = float(zs["dz_max"])
    ratio = float(zs["ratio"]) if zs.get("ratio") is not None else None

    expected = float(slice_thickness) if slice_thickness and slice_thickness > 0 else dz_med

    # big hole in Z
    if dz_max > max(10.0, 5.0 * expected):
        flags.append("missing_slices")

    # nonuniform
    if ratio is not None and ratio > 3.0:
        flags.append("nonuniform_z")

    # PET is naturally coarser; allow slightly higher ratio
    if modality in ("PT", "PET") and "nonuniform_z" in flags and ratio is not None and ratio < 4.5:
        flags.remove("nonuniform_z")

    return flags


def _clean_series_files(series_files: List[Path], eps_mm: float = 0.05) -> Tuple[List[Path], dict]:
    """Return (cleaned_files, meta).

    Removes:
    - mixed acquisitions (different iop/spacing/rows/cols)
    - duplicates at same slice position
    - keeps longest contiguous run
    """
    from collections import Counter

    rows: List[Tuple[Tuple[Any, ...], float, Path]] = []
    for p in series_files:
        try:
            d = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            ipp = np.array(getattr(d, "ImagePositionPatient", [np.nan, np.nan, np.nan]), float)
            iop = np.array(getattr(d, "ImageOrientationPatient", [np.nan] * 6), float)
            ps = getattr(d, "PixelSpacing", None)
            rc = (int(getattr(d, "Rows", -1)), int(getattr(d, "Columns", -1)))
            if not np.isfinite(ipp).all() or not np.isfinite(iop).all() or ps is None:
                continue
            row = iop[:3]
            col = iop[3:]
            n = np.cross(row, col)
            if np.linalg.norm(n) < 1e-6:
                continue
            n = n / np.linalg.norm(n)
            s = float(np.dot(ipp, n))
            key = (tuple(np.round(iop, 4)), rc, tuple(np.round([float(ps[0]), float(ps[1])], 4)))
            rows.append((key, s, p))
        except Exception:
            continue

    if len(rows) < 5:
        return series_files, {
            "clean_status": "too_few_headers",
            "original": len(series_files),
            "kept": len(series_files),
            "dropped": 0,
            "keep_frac": 1.0,
        }

    keys = [r[0] for r in rows]
    best_key, _ = Counter(keys).most_common(1)[0]
    rows = [r for r in rows if r[0] == best_key]

    rows.sort(key=lambda x: x[1])
    svals = np.array([r[1] for r in rows], float)

    # dedup
    keep_idx = [0]
    for i in range(1, len(rows)):
        if abs(svals[i] - svals[keep_idx[-1]]) > eps_mm:
            keep_idx.append(i)
    rows = [rows[i] for i in keep_idx]
    svals = np.array([r[1] for r in rows], float)

    if len(rows) < 5:
        kept = [r[2] for r in rows]
        return kept, {
            "clean_status": "too_short_after_dedup",
            "original": len(series_files),
            "kept": len(kept),
            "dropped": len(series_files) - len(kept),
            "keep_frac": float(len(kept) / max(1, len(series_files))),
        }

    dz = np.diff(svals)
    dz_pos = np.abs(dz[dz != 0])
    dz_med = float(np.median(dz_pos)) if dz_pos.size else float("nan")

    tol = 0.35
    good = np.abs(dz - dz_med) <= tol * max(dz_med, 1e-6)

    # longest run
    best_len = 1
    best_start = 0
    cur_len = 1
    cur_start = 0
    for i, g in enumerate(good, start=1):
        if g:
            cur_len += 1
        else:
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
            cur_start = i
            cur_len = 1
    if cur_len > best_len:
        best_len, best_start = cur_len, cur_start

    rows_run = rows[best_start : best_start + best_len]
    cleaned = [r[2] for r in rows_run]

    meta = {
        "clean_status": "ok",
        "original": len(series_files),
        "kept": len(cleaned),
        "dropped": len(series_files) - len(cleaned),
        "keep_frac": float(len(cleaned) / max(1, len(series_files))),
        "dz_med": dz_med,
        "dz_max": float(np.max(np.abs(np.diff(np.array([r[1] for r in rows_run]))))) if len(rows_run) > 1 else None,
    }
    return cleaned, meta


def _find_dicom_leaf_dirs(patient_dir: Path) -> List[Path]:
    leaf_dirs: List[Path] = []
    for root, _, files in os.walk(patient_dir):
        if not files:
            continue
        # Prefer explicit .dcm
        if any(f.lower().endswith(".dcm") for f in files):
            leaf_dirs.append(Path(root))
            continue
        # If no .dcm: try to detect actual DICOM by reading a couple of files
        if len(files) >= 5:
            sample = None
            for fn in files[:3]:
                p = Path(root) / fn
                if p.is_file():
                    sample = p
                    break
            if sample is not None:
                try:
                    _ = pydicom.dcmread(str(sample), stop_before_pixels=True, force=True)
                    leaf_dirs.append(Path(root))
                except Exception:
                    pass
    return leaf_dirs


def _collect_series(patient_dir: Path) -> Dict[str, List[Path]]:
    """Collect DICOM series using GDCM ordering.

    IMPORTANT: do not sort filenames.
    """
    series_map: Dict[str, List[Path]] = {}
    for d in _find_dicom_leaf_dirs(patient_dir):
        try:
            sids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(d)) or []
            for sid in sids:
                fns = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(d), sid) or []
                if not fns:
                    continue
                ordered = [Path(x) for x in fns]
                if (sid not in series_map) or (len(ordered) > len(series_map[sid])):
                    series_map[sid] = ordered
        except Exception:
            continue
    return series_map


def _make_series_infos(series_map: Dict[str, List[Path]]) -> List[SeriesInfo]:
    infos: List[SeriesInfo] = []
    for sid, files in series_map.items():
        if not files:
            continue
        first = files[0]
        try:
            hdr = _read_header(first)
        except Exception:
            continue
        infos.append(
            SeriesInfo(
                series_uid=sid,
                modality=str(hdr["modality"]),
                series_desc=str(hdr["series_desc"]),
                study_uid=str(hdr["study_uid"]),
                frame_uid=str(hdr["frame_uid"]),
                num_files=len(files),
                first_file=first,
                slice_thickness=hdr["slice_thickness"],
                pixel_spacing=hdr["pixel_spacing"],
                image_type=str(hdr["image_type"]),
                samples_per_pixel=int(hdr["samples_per_pixel"] or 1),
                photometric=str(hdr["photometric"]),
            )
        )
    return infos


# ----------------------------
# Series scoring + selection
# ----------------------------

@dataclass
class SelectionResult:
    ct: Optional[SeriesInfo]
    pet: Optional[SeriesInfo]
    reason: str
    meta: dict


def _build_series_cache(
    infos: List[SeriesInfo],
    series_map: Dict[str, List[Path]],
) -> Dict[str, SeriesCacheEntry]:
    cache: Dict[str, SeriesCacheEntry] = {}
    for info in infos:
        files = series_map.get(info.series_uid, [])
        zs = _series_z_stats(files)
        flags = _geometry_flags(zs, info.slice_thickness, info.modality)
        cleaned, clean_meta = _clean_series_files(files)
        cache[info.series_uid] = SeriesCacheEntry(
            z_stats=zs,
            geom_flags=flags,
            cleaned_files=cleaned,
            clean_meta=clean_meta,
        )
    return cache


def _derived_penalty(image_type: str) -> float:
    it = (image_type or "").upper()
    if "DERIVED" in it or "SECONDARY" in it:
        return -5e4
    return 0.0


def _desc_penalty(desc: str) -> float:
    sd = _safe_lower(desc)
    if any(x in sd for x in ["range", "alpha"]):
        return -1e6
    return 0.0


def _clean_ok(clean_meta: dict, *, min_keep_frac: float = 0.90, min_kept: int = 20) -> Tuple[bool, str]:
    original = int(clean_meta.get("original", 0) or 0)
    kept = int(clean_meta.get("kept", 0) or 0)
    keep_frac = float(clean_meta.get("keep_frac", 0.0) or 0.0)

    # If series is tiny, don't over-penalize.
    if original < 30:
        return True, "ok_small"

    if kept < min_kept:
        return False, f"too_few_kept:{kept}"

    if keep_frac < min_keep_frac:
        return False, f"trimmed_too_much:{kept}/{original}"

    return True, "ok"


def _ct_score(info: SeriesInfo, cache: Dict[str, SeriesCacheEntry], cfg: Config) -> Tuple[float, dict]:
    if info.modality != "CT":
        return -1e18, {}
    if not _is_scalar_series(info):
        return -1e18, {"reject": "non_scalar"}
    if _is_localizer(info, cfg.LOCALIZER_KEYWORDS):
        return -1e18, {"reject": "localizer"}

    c = cache[info.series_uid]
    zs = c.z_stats
    flags = list(c.geom_flags)

    ok_clean, clean_reason = _clean_ok(c.clean_meta)
    if not ok_clean:
        return -1e18, {"reject": f"clean:{clean_reason}", "clean": c.clean_meta, "z": zs, "flags": flags}

    # CT: missing slices is basically always bad for registration & downstream
    if "missing_slices" in flags:
        return -1e18, {"reject": "missing_slices", "clean": c.clean_meta, "z": zs, "flags": flags}

    span = float(zs.get("span_mm") or 0.0)

    score = 0.0
    score += span * 1000.0
    score += float(info.num_files)
    score += _desc_penalty(info.series_desc)
    score += _derived_penalty(info.image_type)

    # Mild preference for reasonable slice thickness
    if info.slice_thickness is not None:
        if info.slice_thickness <= 6.0:
            score += 50
        elif info.slice_thickness >= 8.0:
            score -= 50

    if "nonuniform_z" in flags:
        score -= 1e6
    if "no_z_stats" in flags:
        score -= 1e5

    meta = {"score": score, "z": zs, "flags": flags, "clean": c.clean_meta}
    return score, meta


def _pet_score(info: SeriesInfo, cache: Dict[str, SeriesCacheEntry]) -> Tuple[float, dict]:
    if info.modality not in ("PT", "PET"):
        return -1e18, {}
    if not _is_scalar_series(info):
        return -1e18, {"reject": "non_scalar"}

    c = cache[info.series_uid]
    zs = c.z_stats
    flags = list(c.geom_flags)

    ok_clean, clean_reason = _clean_ok(c.clean_meta)
    if not ok_clean:
        return -1e18, {"reject": f"clean:{clean_reason}", "clean": c.clean_meta, "z": zs, "flags": flags}

    # PET: missing_slices is usually a deal-breaker for SUV space usage
    if "missing_slices" in flags:
        return -1e18, {"reject": "missing_slices", "clean": c.clean_meta, "z": zs, "flags": flags}

    span = float(zs.get("span_mm") or 0.0)

    score = 0.0
    score += span * 1000.0
    score += float(info.num_files)
    score += _desc_penalty(info.series_desc)
    score += _derived_penalty(info.image_type)

    sd = _safe_lower(info.series_desc)
    if "wb" in sd or "whole body" in sd:
        score += 25

    if "nonuniform_z" in flags:
        score -= 5e5
    if "no_z_stats" in flags:
        score -= 1e5

    meta = {"score": score, "z": zs, "flags": flags, "clean": c.clean_meta}
    return score, meta


def _select_series(
    infos: List[SeriesInfo],
    series_map: Dict[str, List[Path]],
    cfg: Config,
) -> SelectionResult:
    """Select PET first, then CT with strict pairing rules."""

    cache = _build_series_cache(infos, series_map)

    cts = [i for i in infos if i.modality == "CT"]
    pets = [i for i in infos if i.modality in ("PT", "PET")]

    debug: dict = {"ct": [], "pet": [], "pair_trials": []}

    # Score candidates
    scored_ct: List[Tuple[SeriesInfo, float, dict]] = []
    for c in cts:
        s, m = _ct_score(c, cache, cfg)
        debug["ct"].append({"series_uid": c.series_uid, "study_uid": c.study_uid, "frame_uid": c.frame_uid,
                            "desc": c.series_desc, "score": s, "meta": m})
        if s > -1e17:
            scored_ct.append((c, s, m))

    scored_pet: List[Tuple[SeriesInfo, float, dict]] = []
    for p in pets:
        s, m = _pet_score(p, cache)
        debug["pet"].append({"series_uid": p.series_uid, "study_uid": p.study_uid, "frame_uid": p.frame_uid,
                             "desc": p.series_desc, "score": s, "meta": m})
        if s > -1e17:
            scored_pet.append((p, s, m))

    scored_ct.sort(key=lambda x: x[1], reverse=True)
    scored_pet.sort(key=lambda x: x[1], reverse=True)

    if not scored_ct:
        return SelectionResult(ct=None, pet=None, reason="no_ct", meta={"debug": debug})

    if not scored_pet:
        # CT-only
        best_ct, _, best_ct_meta = scored_ct[0]
        return SelectionResult(ct=best_ct, pet=None, reason="ct_only_no_pet", meta={"debug": debug, "best": {"ct": best_ct_meta}})

    # Span guard thresholds
    PET_WB_SPAN_MM = 600.0
    CT_MIN_FOR_WB_MM = 350.0

    # Try PET candidates in order; for each, find best matching CT.
    for pet, pet_s, pet_m in scored_pet:
        pet_study = pet.study_uid
        pet_frame = pet.frame_uid
        pet_span = float(pet_m.get("z", {}).get("span_mm") or 0.0)

        # 1) same study CTs
        same_study = [(c, s, m) for (c, s, m) in scored_ct if c.study_uid and c.study_uid == pet_study]

        # 2) same frame CTs (fallback)
        same_frame = []
        if not same_study and pet_frame:
            same_frame = [(c, s, m) for (c, s, m) in scored_ct if c.frame_uid and c.frame_uid == pet_frame]

        pool = same_study if same_study else same_frame
        if not pool:
            debug["pair_trials"].append({
                "pet": pet.series_uid,
                "result": "no_ct_match",
                "pet_study": pet_study,
                "pet_frame": pet_frame,
            })
            continue

        # Prefer frame match inside same-study pool
        if same_study and pet_frame:
            pool_frame = [(c, s, m) for (c, s, m) in pool if c.frame_uid and c.frame_uid == pet_frame]
            if pool_frame:
                pool = pool_frame

        # Apply WB span guard: reject tiny CT for WB PET
        guarded_pool = []
        for c, s, m in pool:
            ct_span = float(m.get("z", {}).get("span_mm") or 0.0)
            if pet_span >= PET_WB_SPAN_MM and ct_span <= CT_MIN_FOR_WB_MM:
                continue
            guarded_pool.append((c, s, m))

        # If guarding removed everything, keep unguarded pool BUT we will mark mismatch later.
        chosen_pool = guarded_pool if guarded_pool else pool
        best_ct, best_ct_s, best_ct_m = max(chosen_pool, key=lambda x: x[1])

        ct_span = float(best_ct_m.get("z", {}).get("span_mm") or 0.0)
        mismatch_span = bool(pet_span >= PET_WB_SPAN_MM and ct_span <= CT_MIN_FOR_WB_MM)

        debug["pair_trials"].append({
            "pet": pet.series_uid,
            "pet_score": pet_s,
            "pet_span": pet_span,
            "ct": best_ct.series_uid,
            "ct_score": best_ct_s,
            "ct_span": ct_span,
            "same_study": bool(best_ct.study_uid == pet_study),
            "same_frame": bool(pet_frame and best_ct.frame_uid == pet_frame),
            "mismatch_span": mismatch_span,
            "guard_used": bool(guarded_pool),
        })

        # Success: paired enough to proceed (study or frame)
        reason = "paired_study" if best_ct.study_uid == pet_study else "paired_frame"
        return SelectionResult(
            ct=best_ct,
            pet=pet,
            reason=reason,
            meta={
                "debug": debug,
                "best": {"ct": best_ct_m, "pet": pet_m},
                "cache_meta": {
                    "ct_clean": cache[best_ct.series_uid].clean_meta,
                    "pet_clean": cache[pet.series_uid].clean_meta,
                },
                "span_guard": {
                    "pet_span_mm": pet_span,
                    "ct_span_mm": ct_span,
                    "pet_wb": pet_span >= PET_WB_SPAN_MM,
                    "ct_too_short_for_wb": ct_span <= CT_MIN_FOR_WB_MM,
                },
            },
        )

    # No PET had a CT match: do CT-only (and do not attempt pet_in_ctspace)
    best_ct, _, best_ct_meta = scored_ct[0]
    best_pet, _, best_pet_meta = scored_pet[0]
    return SelectionResult(
        ct=best_ct,
        pet=best_pet,
        reason="unpaired_pet_ct",
        meta={"debug": debug, "best": {"ct": best_ct_meta, "pet": best_pet_meta}},
    )


# ----------------------------
# IO: read series, force 3D
# ----------------------------

def _read_dicom_series(series_files: List[Path]) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(p) for p in series_files])
    return reader.Execute()


def _force_3d(img: sitk.Image) -> sitk.Image:
    dim = img.GetDimension()
    if dim == 3:
        return img
    if dim == 4:
        size = list(img.GetSize())
        extractor = sitk.ExtractImageFilter()
        extractor.SetSize([size[0], size[1], size[2], 0])
        extractor.SetIndex([0, 0, 0, 0])
        return extractor.Execute(img)
    raise RuntimeError(f"Unsupported image dimension: {dim}")


# ----------------------------
# PET preparation + SUVbw
# ----------------------------

def _prepare_pet_rescaled(pet_img: sitk.Image, pet_dcm_first: Path) -> Tuple[sitk.Image, dict]:
    """SimpleITK/GDCM already applies DICOM rescale when reading most PET datasets.

    We only verify scalar and cast to float32, while recording rescale tags for the manifest.
    """
    d0 = pydicom.dcmread(str(pet_dcm_first), stop_before_pixels=True, force=True)
    try:
        slope0 = float(getattr(d0, "RescaleSlope", 1.0) or 1.0)
    except Exception:
        slope0 = 1.0
    try:
        intercept0 = float(getattr(d0, "RescaleIntercept", 0.0) or 0.0)
    except Exception:
        intercept0 = 0.0

    num_components = pet_img.GetNumberOfComponentsPerPixel()
    if num_components != 1:
        raise RuntimeError(
            f"PET image is not scalar (components={num_components}). "
            "Likely selected a derived/secondary PT series; choose another series."
        )

    pet_rescaled = sitk.Cast(pet_img, sitk.sitkFloat32)
    meta = {
        "rescale_slope": float(slope0),
        "rescale_intercept": float(intercept0),
        "note": "SimpleITK applied rescale automatically; no manual multiplication performed",
    }
    return pet_rescaled, meta


def _parse_hhmmss_to_seconds(s: str) -> Optional[int]:
    if not s:
        return None
    s = str(s).strip()
    if not s:
        return None
    if "." in s:
        s = s.split(".", 1)[0]
    s = s.replace(":", "")
    if len(s) < 6:
        return None
    try:
        hh = int(s[0:2])
        mm = int(s[2:4])
        ss = int(s[4:6])
        return hh * 3600 + mm * 60 + ss
    except Exception:
        return None


def _get_acq_datetime(d: pydicom.Dataset) -> Optional[datetime]:
    adt = getattr(d, "AcquisitionDateTime", None)
    if adt:
        s = str(adt).strip()
        try:
            if "." in s:
                s = s.split(".", 1)[0]
            return datetime.strptime(s, "%Y%m%d%H%M%S")
        except Exception:
            pass

    date = str(getattr(d, "AcquisitionDate", "") or getattr(d, "SeriesDate", "")).strip()
    time = str(getattr(d, "AcquisitionTime", "") or getattr(d, "SeriesTime", "")).strip()
    if date and time:
        tsec = _parse_hhmmss_to_seconds(time)
        if tsec is None:
            return None
        try:
            base = datetime.strptime(date, "%Y%m%d")
            return base.replace(hour=tsec // 3600, minute=(tsec % 3600) // 60, second=tsec % 60)
        except Exception:
            return None
    return None


def _extract_radiopharm_info(d: pydicom.Dataset) -> dict:
    out = {
        "patient_weight_kg": None,
        "dose_bq": None,
        "half_life_s": None,
        "inj_datetime": None,
        "acq_datetime": None,
        "dt_s": None,
        "units": None,
        "corrected_image": None,
        "patient_id": None,
        "study_uid": None,
        "series_uid": None,
    }

    try:
        w = getattr(d, "PatientWeight", None)
        if w is not None:
            out["patient_weight_kg"] = float(w)
    except Exception:
        pass

    try:
        out["units"] = str(getattr(d, "Units", "")).strip().upper() or None
    except Exception:
        pass

    try:
        out["corrected_image"] = _as_str_list(getattr(d, "CorrectedImage", None))
    except Exception:
        out["corrected_image"] = None

    try:
        out["patient_id"] = str(getattr(d, "PatientID", "")).strip() or None
    except Exception:
        pass
    try:
        out["study_uid"] = str(getattr(d, "StudyInstanceUID", "")).strip() or None
    except Exception:
        pass
    try:
        out["series_uid"] = str(getattr(d, "SeriesInstanceUID", "")).strip() or None
    except Exception:
        pass

    out["acq_datetime"] = _get_acq_datetime(d)

    rseq = getattr(d, "RadiopharmaceuticalInformationSequence", None)
    if rseq and len(rseq) > 0:
        r = rseq[0]
        try:
            dose = getattr(r, "RadionuclideTotalDose", None)
            if dose is not None:
                out["dose_bq"] = float(dose)
        except Exception:
            pass
        try:
            hl = getattr(r, "RadionuclideHalfLife", None)
            if hl is not None:
                out["half_life_s"] = float(hl)
        except Exception:
            pass

        inj_dt = getattr(r, "RadiopharmaceuticalStartDateTime", None)
        if inj_dt:
            s = str(inj_dt).strip()
            try:
                if "." in s:
                    s = s.split(".", 1)[0]
                out["inj_datetime"] = datetime.strptime(s, "%Y%m%d%H%M%S")
            except Exception:
                pass
        else:
            inj_date = str(getattr(r, "RadiopharmaceuticalStartDate", "")).strip()
            inj_time = str(getattr(r, "RadiopharmaceuticalStartTime", "")).strip()
            if inj_date and inj_time:
                tsec = _parse_hhmmss_to_seconds(inj_time)
                if tsec is not None:
                    try:
                        base = datetime.strptime(inj_date, "%Y%m%d")
                        out["inj_datetime"] = base.replace(
                            hour=tsec // 3600,
                            minute=(tsec % 3600) // 60,
                            second=tsec % 60,
                        )
                    except Exception:
                        pass

    if out["acq_datetime"] and out["inj_datetime"]:
        out["dt_s"] = float((out["acq_datetime"] - out["inj_datetime"]).total_seconds())

    return out


def _pet_suv_sanity_checks(meta: dict) -> Tuple[bool, str]:
    w = _as_float(meta.get("patient_weight_kg"), None)
    dose = _as_float(meta.get("dose_bq"), None)
    hl = _as_float(meta.get("half_life_s"), None)
    dt = _as_float(meta.get("dt_s"), None)

    if w is None or not (10 <= w <= 300):
        return False, f"bad_weight_kg={w}"
    if dose is None or not (1e6 <= dose <= 5e10):
        return False, f"bad_dose_bq={dose}"
    if hl is None or not (100 <= hl <= 1e6):
        return False, f"bad_half_life_s={hl}"
    if dt is None or not (0 <= dt <= 24 * 3600):
        return False, f"bad_dt_s={dt}"
    return True, "ok"


def _pet_rescaled_to_suvbw(pet_rescaled: sitk.Image, pet_dcm_first: Path) -> Tuple[Optional[sitk.Image], dict]:
    """Convert rescaled PET (activity concentration) to SUVbw when possible.

    Returns (img_or_none, meta).

    Notes:
    - If Units already indicate SUV, returns a copy and sets suv_used=True.
    - If Units are BQML and required tags exist, computes SUVbw.
    - Otherwise returns None with reason.
    """
    d0 = pydicom.dcmread(str(pet_dcm_first), stop_before_pixels=True, force=True)
    meta = _extract_radiopharm_info(d0)

    units = (meta.get("units") or "").upper()
    if units.startswith("SUV"):
        return sitk.Cast(pet_rescaled, sitk.sitkFloat32), {
            **meta,
            "suv_used": True,
            "suv_status": "already_suv",
            "suv_reason": f"units={units}",
        }

    if units not in ("BQML", "BQML ", "BQML\\", "BQML/ML"):
        return None, {
            **meta,
            "suv_used": False,
            "suv_status": "no_suv",
            "suv_reason": f"unsupported_units={units}",
        }

    ok, reason = _pet_suv_sanity_checks(meta)
    if not ok:
        return None, {**meta, "suv_used": False, "suv_status": "no_suv", "suv_reason": reason}

    w = float(meta["patient_weight_kg"])
    dose = float(meta["dose_bq"])
    hl = float(meta["half_life_s"])
    dt = float(meta["dt_s"])

    # decay correction to acquisition time
    decayed_dose = dose * math.exp(-math.log(2.0) * dt / hl)

    if decayed_dose <= 0:
        return None, {**meta, "suv_used": False, "suv_status": "no_suv", "suv_reason": "bad_decayed_dose"}

    # SUVbw = (Bq/mL) * (weight_kg * 1000 g/kg) / (decayed_dose_Bq)
    # Often weight is in kg and Bq/mL; standard SUVbw uses g.
    scale = (w * 1000.0) / decayed_dose

    pet_arr = sitk.GetArrayFromImage(pet_rescaled).astype(np.float32)
    suv_arr = pet_arr * np.float32(scale)

    suv = sitk.GetImageFromArray(suv_arr)
    suv.CopyInformation(pet_rescaled)
    suv = sitk.Cast(suv, sitk.sitkFloat32)

    out_meta = {
        **meta,
        "suv_used": True,
        "suv_status": "computed",
        "suv_reason": "BQML_to_SUVbw",
        "decayed_dose_bq": float(decayed_dose),
        "scale": float(scale),
    }

    # Helpful warning if CorrectedImage doesn't mention decay correction
    corr = _as_str_list(meta.get("corrected_image"))
    if corr and ("DECY" not in corr and "DECAY" not in corr):
        out_meta["warning"] = f"CorrectedImage={corr} (no DECY token)"

    return suv, out_meta


# ----------------------------
# Resampling + registration
# ----------------------------

def _resample_to_reference(moving: sitk.Image, reference: sitk.Image, interp=sitk.sitkLinear) -> sitk.Image:
    return sitk.Resample(moving, reference, sitk.Transform(), interp, 0.0, sitk.sitkFloat32)


def _rigid_register_pet_to_ct(ct: sitk.Image, pet: sitk.Image, seed: int = 42) -> sitk.Transform:
    """Rigid registration PET->CT using Mattes MI.

    Uses rough body masks derived from CT (fixed) and PET (moving) when possible.
    """

    fixed = sitk.Cast(ct, sitk.sitkFloat32)
    moving = sitk.Cast(pet, sitk.sitkFloat32)

    # Fixed mask from CT (rough body)
    fixed_mask = sitk.Cast(fixed > -500.0, sitk.sitkUInt8)

    # Moving mask from PET (rough): threshold to remove background.
    moving_mask = None
    use_moving_mask = True
    try:
        s = sitk.StatisticsImageFilter()
        s.Execute(moving)
        maxv = float(s.GetMaximum())
        meanv = float(s.GetMean())
        sigv = float(s.GetSigma())

        # robust-ish threshold: keeps body while dropping background
        thr = max(0.05 * maxv, meanv + 0.5 * sigv, 1e-6)
        moving_mask = sitk.Cast(moving > thr, sitk.sitkUInt8)

        s2 = sitk.StatisticsImageFilter()
        s2.Execute(moving_mask)
        if s2.GetSum() < 1000:
            use_moving_mask = False
    except Exception:
        use_moving_mask = False
        moving_mask = None


    init = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=80)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, seed=seed)
    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=300,
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-8,
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Masks
    try:
        reg.SetMetricFixedMask(fixed_mask)
    except Exception:
        pass
    if use_moving_mask:
        try:
            reg.SetMetricMovingMask(moving_mask)
        except Exception:
            pass

    reg.SetInitialTransform(init, inPlace=False)
    return reg.Execute(fixed, moving)


# ----------------------------
# PET/CT QC
# ----------------------------

def _compute_pet_ct_qc(pet_suv_in_ct: sitk.Image, ct: sitk.Image, out_hot_thr: float = 50.0, in_hot_thr: float = 80.0) -> dict:
    """Lightweight QC to catch obvious mis-registration / wrong FOV.

    - Body mask: CT > -500 HU
    - Look at SUV distribution in body
    - Hot voxels outside body
    """

    pet = sitk.Cast(pet_suv_in_ct, sitk.sitkFloat32)
    ct_f = sitk.Cast(ct, sitk.sitkFloat32)

    body = sitk.Cast(ct_f > -500.0, sitk.sitkUInt8)

    pet_arr = sitk.GetArrayFromImage(pet).astype(np.float32)
    body_arr = sitk.GetArrayFromImage(body).astype(np.uint8)

    inside = pet_arr[body_arr > 0]
    outside = pet_arr[body_arr == 0]

    qc: dict = {"qc_ok": False}

    if inside.size < 1000:
        qc.update({"qc_reason": "too_small_body_mask", "in_body_vox": int(inside.size)})
        return qc

    in_p99 = float(np.percentile(inside, 99))
    in_max = float(np.max(inside))

    out_hot = outside[outside >= out_hot_thr]
    out_hot_frac = float(out_hot.size / max(1, outside.size))

    in_hot = inside[inside >= in_hot_thr]
    in_hot_frac = float(in_hot.size / max(1, inside.size))

    qc.update(
        {
            "in_body_p99": in_p99,
            "in_body_max": in_max,
            "out_hot_frac": out_hot_frac,
            "in_hot80_frac": in_hot_frac,
            "in_hot80_vox": int(in_hot.size),
        }
    )

    # Heuristic decision: tuned to catch the "PET is out of body" failure mode.
    # - if in_body_p99 is extremely low, PET probably landed outside body.
    # - if there's a lot of very hot activity outside body, likely misaligned.
    ok = True
    reasons: List[str] = []

    if in_p99 <= 0.25:
        ok = False
        reasons.append("in_body_p99_too_low")

    if out_hot_frac >= 0.002:
        ok = False
        reasons.append("too_hot_outside")

    qc["qc_ok"] = bool(ok)
    qc["qc_reason"] = "ok" if ok else "|".join(reasons)
    return qc


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--patients", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--register-pet", action="store_true", help="Rigid-register PET to CT space.")
    parser.add_argument("--no-suv", action="store_true", help="Skip SUV conversion.")
    parser.add_argument("--save-pet-raw", action="store_true", help="Also save aligned rescaled PET for debugging.")
    args = parser.parse_args()

    cfg = Config()
    ensure_project_dirs(cfg)

    verify_write_access(cfg.DATA_ROOT)

    if not cfg.DICOM_ROOT.exists():
        raise FileNotFoundError(f"DICOM_ROOT not found: {cfg.DICOM_ROOT}")

    patient_dirs = sorted([
    p for p in cfg.DICOM_ROOT.iterdir()
    if p.is_dir() and (
        p.name.startswith("Lung_Dx-") or
        (len(p.name) >= 2 and p.name[0].isalpha() and p.name[1:].isdigit())
    )
    ])

    if args.patients.strip():
        wanted = {x.strip() for x in args.patients.split(",") if x.strip()}
        patient_dirs = [p for p in patient_dirs if p.name in wanted]

    if args.limit and args.limit > 0:
        patient_dirs = patient_dirs[: args.limit]

    rows: List[dict] = []

    index_path = cfg.DATA_DIR / "index_converted.csv"
    existing_index = _load_index(index_path)

    for pat_dir in tqdm(patient_dirs, desc="Convert+SUV+Register"):
        patient_id = pat_dir.name

        out_dir = cfg.CASES_DIR / patient_id / "nifti"
        out_ct = out_dir / "ct.nii.gz"
        out_pet = out_dir / "pet.nii.gz"  # canonical only if suv+qc pass
        out_pet_suv_ct = out_dir / "pet_suv_ctspace.nii.gz"
        out_pet_native_suv = out_dir / "pet_native_suv.nii.gz"
        out_pet_native_rescaled = out_dir / "pet_native_rescaled.nii.gz"
        out_pet_raw_ct = out_dir / "pet_rescaled_raw_ctspace.nii.gz"
        out_tx = out_dir / "pet_to_ct.tfm"

        manifest: dict = {
            "patient_id": patient_id,
            "status": "started",
            "paths": {
                "ct": str(out_ct),
                "pet": str(out_pet),
                "pet_suv_ct": str(out_pet_suv_ct),
                "pet_native_suv": str(out_pet_native_suv),
                "pet_native_rescaled": str(out_pet_native_rescaled),
                "pet_raw_ct": str(out_pet_raw_ct),
                "pet_to_ct_transform": str(out_tx),
            },
        }

        # Skip logic: if CT exists and we already have a finalized manifest status, skip unless overwrite.
        manifest_path = (cfg.CASES_DIR / patient_id / "manifest.json")
        if not args.overwrite and out_ct.exists() and manifest_path.exists():
            try:
                prev = json.loads(manifest_path.read_text())
                prev_status = str(prev.get("status", ""))
                if prev_status in ("ok", "ct_only_no_pet", "ct_only", "ct_ok_pet_mismatch", "ct_ok_pet_partial"):
                    rows.append({
                        "patient_id": patient_id,
                        "has_ct": True,
                        "has_pet": bool((cfg.CASES_DIR / patient_id / "nifti" / "pet.nii.gz").exists()),
                        "has_pet_suv": bool(out_pet_suv_ct.exists() or out_pet_native_suv.exists()),
                        "qc_pass": bool(prev.get("qc", {}).get("qc_ok", False)),
                        "ct_path": str(out_ct),
                        "pet_path": str(out_pet) if out_pet.exists() else "",
                        "note": f"skipped_existing:{prev_status}",
                    })
                    continue
            except Exception:
                pass

        # Discover and select
        series_map = _collect_series(pat_dir)
        infos = _make_series_infos(series_map)
        sel = _select_series(infos, series_map, cfg)

        manifest["selection"] = {
            "ct_series_uid": sel.ct.series_uid if sel.ct else None,
            "ct_study_uid": sel.ct.study_uid if sel.ct else None,
            "ct_frame_uid": sel.ct.frame_uid if sel.ct else None,
            "pet_series_uid": sel.pet.series_uid if sel.pet else None,
            "pet_study_uid": sel.pet.study_uid if sel.pet else None,
            "pet_frame_uid": sel.pet.frame_uid if sel.pet else None,
            "selection_reason": sel.reason,
            "study_pair": bool(sel.ct and sel.pet and sel.ct.study_uid and sel.ct.study_uid == sel.pet.study_uid),
            "frame_pair": bool(sel.ct and sel.pet and sel.ct.frame_uid and sel.ct.frame_uid == sel.pet.frame_uid),
        }
        manifest["selection_debug"] = sel.meta.get("debug", {})
        manifest["span_guard"] = sel.meta.get("span_guard", {})

        if sel.ct is None:
            manifest["status"] = "failed"
            manifest["error"] = "no_ct_series_found"
            _save_manifest(cfg.CASES_DIR / patient_id, manifest)
            rows.append({
                "patient_id": patient_id,
                "has_ct": False,
                "has_pet": False,
                "has_pet_suv": False,
                "qc_pass": False,
                "ct_path": "",
                "pet_path": "",
                "note": "no_ct_series_found",
            })
            continue

        # Read CT using cleaned files from cache (recompute quickly here for safety)
        ct_files_clean, ct_clean_meta = _clean_series_files(series_map[sel.ct.series_uid])
        manifest["ct_clean"] = ct_clean_meta

        try:
            ct_img = _force_3d(_read_dicom_series(ct_files_clean))
            _write_nifti(ct_img, out_ct)
            manifest["status"] = "ct_ok"
        except Exception as e:
            manifest["status"] = "failed"
            manifest["error"] = f"ct_convert_failed:{repr(e)}"
            _save_manifest(cfg.CASES_DIR / patient_id, manifest)
            rows.append({
                "patient_id": patient_id,
                "has_ct": False,
                "has_pet": False,
                "has_pet_suv": False,
                "qc_pass": False,
                "ct_path": "",
                "pet_path": "",
                "note": f"ct_convert_failed:{repr(e)}",
            })
            continue

        # If no PET selected or unpaired, do CT-only and stop (no garbage PET-in-CT-space)
        if sel.pet is None:
            manifest["status"] = "ct_only_no_pet"
            manifest["note"] = "no_pet_series_found"
            _save_manifest(cfg.CASES_DIR / patient_id, manifest)
            rows.append({
                "patient_id": patient_id,
                "has_ct": True,
                "has_pet": False,
                "has_pet_suv": False,
                "qc_pass": False,
                "ct_path": str(out_ct),
                "pet_path": "",
                "note": "no_pet_series_found",
            })
            continue

        paired = manifest["selection"]["study_pair"] or manifest["selection"]["frame_pair"]
        if not paired:
            # Still save PET native outputs if possible, but do not register/resample.
            manifest["note"] = "pet_ct_unpaired_skip_ctspace"

        # Read PET
        pet_files_clean, pet_clean_meta = _clean_series_files(series_map[sel.pet.series_uid])
        manifest["pet_clean"] = pet_clean_meta

        try:
            pet_img = _force_3d(_read_dicom_series(pet_files_clean))
            pet_rescaled, rescale_meta = _prepare_pet_rescaled(pet_img, sel.pet.first_file)
            manifest["pet_rescale"] = rescale_meta
            _write_nifti(pet_rescaled, out_pet_native_rescaled)

            if args.no_suv:
                pet_suv = None
                suv_meta = {"suv_used": False, "suv_status": "skipped", "suv_reason": "--no-suv"}
            else:
                pet_suv, suv_meta = _pet_rescaled_to_suvbw(pet_rescaled, sel.pet.first_file)
                if pet_suv is not None and suv_meta.get("suv_used"):
                    _write_nifti(pet_suv, out_pet_native_suv)

            manifest["suv"] = suv_meta

        except Exception as e:
            manifest["status"] = "failed"
            manifest["error"] = f"pet_read_failed:{repr(e)}"
            _save_manifest(cfg.CASES_DIR / patient_id, manifest)
            rows.append({
                "patient_id": patient_id,
                "has_ct": True,
                "has_pet": False,
                "has_pet_suv": False,
                "qc_pass": False,
                "ct_path": str(out_ct),
                "pet_path": "",
                "note": f"pet_read_failed:{repr(e)}",
            })
            continue

        # If we can't compute SUV, we can still stop here.
        if not (manifest.get("suv", {}).get("suv_used") and out_pet_native_suv.exists()):
            manifest["status"] = "ct_ok_pet_partial"
            manifest["note"] = f"no_suv:{manifest.get('suv', {}).get('suv_reason')}"
            _save_manifest(cfg.CASES_DIR / patient_id, manifest)
            rows.append({
                "patient_id": patient_id,
                "has_ct": True,
                "has_pet": False,
                "has_pet_suv": False,
                "qc_pass": False,
                "ct_path": str(out_ct),
                "pet_path": "",
                "note": f"no_suv:{manifest.get('suv', {}).get('suv_reason')}"
            })
            continue

        # Load SUV image from disk to ensure we use the saved object
        pet_suv = sitk.ReadImage(str(out_pet_native_suv), sitk.sitkFloat32)

        # Span mismatch guard: if WB PET + tiny CT, conditionally allow ctspace outputs
        # Keep `paired` to ensure we only trigger this for valid CT/PET pairings
        span_guard = manifest.get("span_guard", {})
        skip_pet_ctspace = False  # default: allow
        ctspace_note = None

        if paired and span_guard.get("pet_wb") and span_guard.get("ct_too_short_for_wb"):
            # New behavior:
            # - If frame_pair is True: allow generating ct-space PET (thorax ROI is valid)
            # - If frame_pair is False: defer decision until after registration + QC
            if manifest["selection"].get("frame_pair", False):
                skip_pet_ctspace = False
                ctspace_note = "thorax_only_from_wb_pet"
            else:
                # Conditional: will be decided after registration + QC
                skip_pet_ctspace = None  # means "decide later based on tx + qc"

        if not paired:
            manifest["status"] = "ct_ok_pet_mismatch"
            manifest["note"] = "unpaired_pet_ct_skip_ctspace"
            _save_manifest(cfg.CASES_DIR / patient_id, manifest)
            rows.append({
                "patient_id": patient_id,
                "has_ct": True,
                "has_pet": False,
                "has_pet_suv": True,
                "qc_pass": False,
                "ct_path": str(out_ct),
                "pet_path": "",
                "note": "unpaired_pet_ct_skip_ctspace",
            })
            continue

        # Register/resample
        try:
            tx = None  # Track whether registration produced a transform
            if args.register_pet:
                tx = _rigid_register_pet_to_ct(ct_img, pet_suv, seed=42)
                sitk.WriteTransform(tx, str(out_tx))
                pet_suv_in_ct = sitk.Resample(pet_suv, ct_img, tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
                pet_rescaled_native = sitk.ReadImage(str(out_pet_native_rescaled), sitk.sitkFloat32)
                pet_rescaled_in_ct = sitk.Resample(pet_rescaled_native, ct_img, tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
            else:
                # No registration: identity resample
                pet_suv_in_ct = _resample_to_reference(pet_suv, ct_img, interp=sitk.sitkLinear)
                pet_rescaled_native = sitk.ReadImage(str(out_pet_native_rescaled), sitk.sitkFloat32)
                pet_rescaled_in_ct = _resample_to_reference(pet_rescaled_native, ct_img, interp=sitk.sitkLinear)

            if args.save_pet_raw:
                _write_nifti(pet_rescaled_in_ct, out_pet_raw_ct)

            # Compute QC first (before deciding whether to write)
            qc = _compute_pet_ct_qc(pet_suv_in_ct, ct_img, out_hot_thr=50.0, in_hot_thr=80.0)
            manifest["qc"] = qc

            # Decide whether to write pet_suv_ctspace based on skip_pet_ctspace, tx, and QC
            allow_write_ctspace = True

            if skip_pet_ctspace is True:
                allow_write_ctspace = False
            elif skip_pet_ctspace is None:
                # Deferred decision: require good registration AND passing QC
                if args.register_pet:
                    allow_write_ctspace = bool((tx is not None) and qc.get("qc_ok", False))
                else:
                    # Identity resample when frame_pair=False is NOT allowed
                    # (frame_pair=False + no registration is too risky)
                    allow_write_ctspace = False
                    manifest.setdefault("output_meta", {})["ctspace_skipped_reason"] = \
                        "frame_pair_false_and_registration_disabled"

            if allow_write_ctspace:
                _write_nifti(pet_suv_in_ct, out_pet_suv_ct)

                # Add ctspace_note if this is WB PET -> small CT case
                if paired and span_guard.get("pet_wb") and span_guard.get("ct_too_short_for_wb"):
                    manifest.setdefault("output_meta", {})["ctspace_note"] = "thorax_only_from_wb_pet"

                if qc.get("qc_ok", False):
                    shutil.copyfile(str(out_pet_suv_ct), str(out_pet))
                    manifest["status"] = "ok"
                    _save_manifest(cfg.CASES_DIR / patient_id, manifest)
                    rows.append({
                        "patient_id": patient_id,
                        "has_ct": True,
                        "has_pet": True,
                        "has_pet_suv": True,
                        "qc_pass": True,
                        "qc_in_body_p99": qc.get("in_body_p99"),
                        "qc_in_body_max": qc.get("in_body_max"),
                        "qc_hot_out_frac": qc.get("out_hot_frac"),
                        "qc_in_hot80_frac": qc.get("in_hot80_frac"),
                        "ct_path": str(out_ct),
                        "pet_path": str(out_pet),
                        "ct_series_uid": sel.ct.series_uid,
                        "pet_series_uid": sel.pet.series_uid,
                        "note": f"paired={'study' if manifest['selection']['study_pair'] else 'frame'}|qc=ok",
                    })
                else:
                    # ctspace written but QC failed: canonical pet.nii.gz not created
                    manifest["status"] = "ct_ok_pet_partial"
                    manifest["note"] = f"qc_failed:{qc.get('qc_reason')}"
                    _save_manifest(cfg.CASES_DIR / patient_id, manifest)
                    rows.append({
                        "patient_id": patient_id,
                        "has_ct": True,
                        "has_pet": False,  # canonical pet.nii.gz not created
                        "has_pet_suv": True,  # pet_suv_ctspace.nii.gz exists
                        "qc_pass": False,
                        "qc_in_body_p99": qc.get("in_body_p99"),
                        "qc_in_body_max": qc.get("in_body_max"),
                        "qc_hot_out_frac": qc.get("out_hot_frac"),
                        "qc_in_hot80_frac": qc.get("in_hot80_frac"),
                        "ct_path": str(out_ct),
                        "pet_path": "",
                        "ct_series_uid": sel.ct.series_uid,
                        "pet_series_uid": sel.pet.series_uid,
                        "note": f"paired={'study' if manifest['selection']['study_pair'] else 'frame'}|qc={qc.get('qc_reason')}",
                    })
            else:
                # Deferred decision resulted in skip: delete stale outputs if they exist
                stale_paths = [out_pet_suv_ct, out_pet]
                if args.save_pet_raw:
                    stale_paths.append(out_pet_raw_ct)
                for stale_path in stale_paths:
                    if stale_path.exists():
                        try:
                            stale_path.unlink()
                        except Exception:
                            pass

                manifest["status"] = "ct_ok_pet_mismatch"
                # Use the reason already set, or default to unreliable registration
                if "ctspace_skipped_reason" not in manifest.get("output_meta", {}):
                    manifest.setdefault("output_meta", {})["ctspace_skipped_reason"] = \
                        "wb_pet_vs_small_ct_unreliable_registration"
                manifest["note"] = manifest.get("output_meta", {}).get("ctspace_skipped_reason", "skip_ctspace")
                _save_manifest(cfg.CASES_DIR / patient_id, manifest)
                rows.append({
                    "patient_id": patient_id,
                    "has_ct": True,
                    "has_pet": False,
                    "has_pet_suv": True,  # native SUV still exists
                    "qc_pass": False,
                    "ct_path": str(out_ct),
                    "pet_path": "",
                    "ct_series_uid": sel.ct.series_uid,
                    "pet_series_uid": sel.pet.series_uid,
                    "note": manifest["note"],
                })

        except Exception as e:
            manifest["status"] = "failed"
            manifest["error"] = f"register_or_resample_failed:{repr(e)}"
            _save_manifest(cfg.CASES_DIR / patient_id, manifest)
            rows.append({
                "patient_id": patient_id,
                "has_ct": True,
                "has_pet": False,
                "has_pet_suv": True,
                "qc_pass": False,
                "ct_path": str(out_ct),
                "pet_path": "",
                "ct_series_uid": sel.ct.series_uid,
                "pet_series_uid": sel.pet.series_uid,
                "note": f"register_or_resample_failed:{repr(e)}",
            })

    df = pd.DataFrame(rows)
    merged = _merge_index(existing_index, df)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(index_path, index=False)

    print(f"\nWrote index: {index_path}")
    if not merged.empty:
        cols = [c for c in ["patient_id", "has_ct", "has_pet", "has_pet_suv", "qc_pass", "note"] if c in merged.columns]
        print(merged[cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
