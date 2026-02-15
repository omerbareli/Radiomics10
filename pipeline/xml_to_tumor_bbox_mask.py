# pipeline/xml_to_tumor_bbox_mask.py
"""
running with shrinking and windowing parameters on specific patient:
python -m pipeline.xml_to_tumor_bbox_mask \
  --use-medsam2 \
  --patients Lung_Dx-A0165 \
  --overwrite \
  --lung-mask-path auto \
  --window-lower -1024 \
  --window-upper 400 \
  --shrink-frac 0.20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from tqdm import tqdm

from pipeline.config import Config, ensure_project_dirs, verify_write_access


UID_RE = re.compile(r"\b\d+\.\d+(?:\.\d+)+\b")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def _find_dicom_leaf_dirs(patient_dir: Path) -> List[Path]:
    """Dirs containing at least one .dcm file (non-recursive series reader needs dirs)."""
    leaf_dirs: List[Path] = []
    for root, _, files in os.walk(patient_dir):
        if any(f.lower().endswith(".dcm") for f in files):
            leaf_dirs.append(Path(root))
    return leaf_dirs


def _get_series_files_by_uid(patient_dir: Path, series_uid: str) -> List[Path]:
    """
    Locate DICOM files for a specific SeriesInstanceUID using GDCM series discovery.

    Robustness improvement:
    - accumulate files across *all* leaf dirs containing that SeriesInstanceUID,
      because some datasets split a series across multiple folders.
    """
    found: List[Path] = []
    for d in _find_dicom_leaf_dirs(patient_dir):
        try:
            sids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(d))
            if not sids:
                continue
            if series_uid in sids:
                fns = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(d), series_uid)
                found.extend(Path(x) for x in fns)
        except Exception:
            continue

    # De-dup while preserving a stable order
    if not found:
        return []
    uniq = list(dict.fromkeys(str(p) for p in found).keys())
    return [Path(x) for x in uniq]


def _read_dicom_header(fp: Path) -> Optional[pydicom.Dataset]:
    try:
        return pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
    except Exception:
        return None


def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _compute_slice_sort_key(ds: pydicom.Dataset) -> Tuple[int, float, int]:
    """
    Return a robust sortable key for slice ordering.

    Preferred:
    - Use ImageOrientationPatient to compute the slice normal and
      project ImagePositionPatient onto that normal (robust for oblique scans).

    Fallback:
    - Use ImagePositionPatient[2]
    - Else InstanceNumber
    """
    # InstanceNumber fallback (stable-ish)
    inst = getattr(ds, "InstanceNumber", None)
    inst_i = 0
    try:
        inst_i = int(inst) if inst is not None else 0
    except Exception:
        inst_i = 0

    ipp = getattr(ds, "ImagePositionPatient", None)
    iop = getattr(ds, "ImageOrientationPatient", None)

    # If we have IOP + IPP, compute scalar position along slice normal
    if ipp is not None and len(ipp) >= 3 and iop is not None and len(iop) >= 6:
        try:
            row = np.array([float(iop[0]), float(iop[1]), float(iop[2])], dtype=float)
            col = np.array([float(iop[3]), float(iop[4]), float(iop[5])], dtype=float)
            normal = np.cross(row, col)
            nrm = float(np.linalg.norm(normal))
            if nrm > 0:
                normal /= nrm
                pos = np.array([float(ipp[0]), float(ipp[1]), float(ipp[2])], dtype=float)
                s = float(np.dot(pos, normal))
                return (0, s, inst_i)
        except Exception:
            pass

    # Fallback: z coordinate if present
    if ipp is not None and len(ipp) >= 3:
        z = _safe_float(ipp[2])
        if z is not None:
            return (1, z, inst_i)

    # Final fallback: InstanceNumber only
    return (2, 0.0, inst_i)


def _build_sopuid_to_slice_index(
    ct_img: sitk.Image,
    series_files: List[Path],
) -> Tuple[Dict[str, int], dict]:
    """
    Build SOPInstanceUID -> z-index mapping for CT series files.

    Returns:
      (sop2z, stats)
    """
    records: List[Tuple[str, Tuple[int, float, int]]] = []
    readable = 0
    missing_sop = 0

    for fp in series_files:
        ds = _read_dicom_header(fp)
        if ds is None:
            continue
        readable += 1

        sop = str(getattr(ds, "SOPInstanceUID", "")).strip()
        if not sop:
            missing_sop += 1
            continue

        key = _compute_slice_sort_key(ds)
        records.append((sop, key))

    if not records:
        return {}, {
            "readable_headers": readable,
            "records": 0,
            "missing_sop_in_headers": missing_sop,
            "mapping_suspect": True,
            "reason": "no records",
        }

    # Sort by physical order key; stable tie-break by SOP string
    records.sort(key=lambda t: (t[1][0], t[1][1], t[1][2], t[0]))

    sop2z: Dict[str, int] = {}
    for idx, (sop, _) in enumerate(records):
        sop2z[sop] = idx

    zdim = int(ct_img.GetSize()[2])
    nmap = len(sop2z)

    # Sanity checks: mapping should be close to ct zdim
    mapping_suspect = False
    reason = ""

    # If we are way off, it's likely not the same slice set used to make ct.nii.gz
    if zdim > 0:
        if nmap < int(0.90 * zdim) and (zdim - nmap) > 5:
            mapping_suspect = True
            reason = f"mapped_slices_too_few (mapped={nmap}, ct_zdim={zdim})"
        elif nmap > int(1.10 * zdim) and (nmap - zdim) > 5:
            mapping_suspect = True
            reason = f"mapped_slices_too_many (mapped={nmap}, ct_zdim={zdim})"

    return sop2z, {
        "readable_headers": readable,
        "records": len(records),
        "missing_sop_in_headers": missing_sop,
        "mapped_slices": nmap,
        "ct_zdim": zdim,
        "mapping_suspect": mapping_suspect,
        "reason": reason,
    }


def _parse_bboxes_from_xml(xml_path: Path) -> List[Tuple[int, int, int, int]]:
    """
    Parse Pascal/VOC-like structure:
      <object><bndbox><xmin>..</xmin><ymin>..</ymin><xmax>..</xmax><ymax>..</ymax></bndbox></object>
    """
    root = ET.parse(str(xml_path)).getroot()
    bboxes: List[Tuple[int, int, int, int]] = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            xmin = int(float(bb.findtext("xmin", "0")))
            ymin = int(float(bb.findtext("ymin", "0")))
            xmax = int(float(bb.findtext("xmax", "0")))
            ymax = int(float(bb.findtext("ymax", "0")))
            bboxes.append((xmin, ymin, xmax, ymax))
        except Exception:
            continue
    return bboxes


def _shrink_bbox(
    xmin: int, ymin: int, xmax: int, ymax: int,
    shrink_frac: float,
    min_side_px: int,
) -> Optional[Tuple[int, int, int, int]]:
    if xmax < xmin or ymax < ymin:
        return None

    w = xmax - xmin + 1
    h = ymax - ymin + 1
    if w <= 0 or h <= 0:
        return None

    dx = int(round(w * shrink_frac / 2.0))
    dy = int(round(h * shrink_frac / 2.0))

    xmin2, xmax2 = xmin + dx, xmax - dx
    ymin2, ymax2 = ymin + dy, ymax - dy

    w2 = xmax2 - xmin2 + 1
    h2 = ymax2 - ymin2 + 1
    if w2 < min_side_px or h2 < min_side_px:
        return None
    if xmax2 < xmin2 or ymax2 < ymin2:
        return None
    return xmin2, ymin2, xmax2, ymax2


def _resolve_annotation_dir(cfg: Config, pid: str) -> Optional[Path]:
    """
    Try a few common mappings. Keeps your A/B/E/G#### logic, plus fallbacks.
    """
    m = re.search(r"([ABEG]\d{4})", pid)
    candidates: List[Path] = []
    if m:
        candidates.append(cfg.ANNOTATION_ROOT / m.group(1))

    tail = pid.split("-")[-1]
    candidates += [
        cfg.ANNOTATION_ROOT / tail,
        cfg.ANNOTATION_ROOT / pid,
    ]

    for c in candidates:
        if c.exists():
            return c
    return None


def _pick_sop_for_xml(xml_path: Path, sop2z: Dict[str, int]) -> Optional[str]:
    """
    Robust SOP selection:
    1) if filename contains a UID candidate that exists in sop2z, use it
    2) otherwise, scan XML text for all UID candidates and pick first that exists in sop2z
    """
    # Filename candidates (more robust than stem==uid)
    name_uids = UID_RE.findall(xml_path.name)
    for u in name_uids:
        if u in sop2z:
            return u

    # Text candidates
    try:
        txt = xml_path.read_text(errors="ignore")
    except Exception:
        return None

    uids = UID_RE.findall(txt)
    for u in uids:
        if u in sop2z:
            return u

    return None


def _detect_one_based(xml_files: List[Path]) -> bool:
    """
    Conservative 1-based detector:
    - sample up to 50 xmls
    - if we see *many* xmin/ymin == 1 and *none* == 0, assume 1-based
    """
    mins: List[Tuple[int, int]] = []
    for xf in xml_files[: min(50, len(xml_files))]:
        for (xmin, ymin, xmax, ymax) in _parse_bboxes_from_xml(xf):
            mins.append((xmin, ymin))

    if not mins:
        return False

    xs = [m[0] for m in mins]
    ys = [m[1] for m in mins]
    count0 = sum(1 for v in xs + ys if v == 0)
    count1 = sum(1 for v in xs + ys if v == 1)

    # Require: no zeros AND at least 20% are ones (stronger signal)
    if count0 == 0 and count1 >= max(3, int(0.20 * len(xs + ys))):
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing tumor_bbox_mask.nii.gz per case.")
    parser.add_argument("--shrink-frac", type=float, default=0.15, help="Shrink bbox by this fraction (0.10-0.20 typical).")
    parser.add_argument("--min-side-px", type=int, default=8, help="Drop bboxes that shrink below this size.")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N patients (0=all).")
    parser.add_argument("--patients", type=str, default="", help="Comma-separated patient IDs to process.")
    parser.add_argument("--auto-one-based", action="store_true", help="If bbox coords look 1-based, subtract 1 (best-effort).")
    parser.add_argument(
        "--fail-on-mapping-suspect",
        action="store_true",
        help="If SOP->slice mapping looks inconsistent with CT z-dimension, fail the case (safer).",
    )
    # MedSAM2 arguments
    parser.add_argument(
        "--use-medsam2",
        action="store_true",
        help="Use MedSAM2 for lesion segmentation instead of rectangular bboxes.",
    )
    parser.add_argument(
        "--medsam2-checkpoint",
        type=str,
        default=None,
        help="Path to MedSAM2 checkpoint (overrides config default).",
    )
    parser.add_argument(
        "--window-lower",
        type=float,
        default=-1024,
        help="CT window lower bound in HU for MedSAM2 preprocessing.",
    )
    parser.add_argument(
        "--window-upper",
        type=float,
        default=400,
        help="CT window upper bound in HU for MedSAM2 preprocessing.",
    )
    parser.add_argument(
        "--lung-mask-path",
        type=str,
        default=None,
        help="Optional path to lung mask NIfTI to constrain tumor to lung region. If 'auto', reads from manifest.",
    )
    args = parser.parse_args()

    cfg = Config()
    ensure_project_dirs(cfg)

    # Pre-flight permission check on mounted data directory
    verify_write_access(cfg.DATA_ROOT)

    index_csv = cfg.DATA_DIR / "index_converted.csv"
    if not index_csv.exists():
        raise FileNotFoundError(f"index_converted.csv not found at: {index_csv}")

    df = pd.read_csv(index_csv)

    if args.patients.strip():
        wanted = {x.strip() for x in args.patients.split(",") if x.strip()}
        df = df[df["patient_id"].astype(str).isin(wanted)].copy()

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    rows: List[dict] = []
    out_index = cfg.DATA_DIR / "index_tumor_bbox_masks.csv"

    # Initialize MedSAM2 segmenter if enabled
    medsam2_segmenter = None
    segmentation_method = "bbox"
    if args.use_medsam2:
        from pipeline.medsam2_inference import MedSAM2Segmenter
        checkpoint = Path(args.medsam2_checkpoint) if args.medsam2_checkpoint else cfg.MEDSAM2_CHECKPOINT
        medsam2_segmenter = MedSAM2Segmenter(
            checkpoint_path=checkpoint,
            config_path=cfg.MEDSAM2_CONFIG,
        )
        segmentation_method = "medsam2"
        print(f"MedSAM2 enabled. Checkpoint: {checkpoint}")

    for _, r in tqdm(df.iterrows(), total=len(df), desc="XML->tumor bbox mask"):
        pid = str(r["patient_id"])

        # --- case paths ---
        case_dir = cfg.CASES_DIR / pid
        seg_dir = case_dir / "seg"
        # Use different output filename for MedSAM2 vs bbox mode
        if args.use_medsam2:
            out_mask_path = seg_dir / "tumor_mask_fm.nii.gz"  # fm = foundation model
            manifest_key = "tumor_mask_fm"
        else:
            out_mask_path = seg_dir / "tumor_bbox_mask.nii.gz"
            manifest_key = "tumor_bbox_mask"
        manifest_path = case_dir / "manifest.json"

        t0 = time.time()
        status = "ok"
        err = ""

        used = 0
        missing_sop = 0
        skipped_small = 0
        applied_one_based = False
        mapping_stats: dict = {}

        ct_path = Path(str(r["ct_path"]))
        if not ct_path.exists():
            rows.append({"patient_id": pid, "status": "missing_ct", "error": str(ct_path)})
            continue

        if out_mask_path.exists() and not args.overwrite:
            rows.append({"patient_id": pid, "status": "skipped_exists", "tumor_bbox_mask_path": str(out_mask_path), "error": ""})
            continue

        ann_dir = _resolve_annotation_dir(cfg, pid)
        if ann_dir is None:
            rows.append({"patient_id": pid, "status": "missing_annotations_dir", "error": f"Could not resolve ann dir under {cfg.ANNOTATION_ROOT}"})
            continue

        xml_files = sorted(ann_dir.glob("*.xml"))
        if not xml_files:
            rows.append({"patient_id": pid, "status": "missing_xml_files", "error": str(ann_dir)})
            continue

        ct_series_uid = str(r.get("ct_series_uid", "") or "").strip()
        if not ct_series_uid or ct_series_uid.lower() == "nan":
            # Fallback: read from manifest.json (always has ct_series_uid)
            manifest = _load_json(manifest_path)
            ct_series_uid = manifest.get("selection", {}).get("ct_series_uid", "")
            if not ct_series_uid:
                rows.append({"patient_id": pid, "status": "missing_ct_series_uid", "error": "ct_series_uid not in index or manifest"})
                continue

        dicom_patient_dir = cfg.DICOM_ROOT / pid
        if not dicom_patient_dir.exists():
            rows.append({"patient_id": pid, "status": "missing_dicom_patient_dir", "error": str(dicom_patient_dir)})
            continue

        try:
            series_files = _get_series_files_by_uid(dicom_patient_dir, ct_series_uid)
            if not series_files:
                raise RuntimeError(f"Could not locate CT DICOM series files for SeriesInstanceUID={ct_series_uid}")

            ct_img = sitk.ReadImage(str(ct_path))
            sop2z, mapping_stats = _build_sopuid_to_slice_index(ct_img, series_files)
            if not sop2z:
                raise RuntimeError("Failed to build SOP->slice mapping (no readable CT headers?)")

            if mapping_stats.get("mapping_suspect", False) and args.fail_on_mapping_suspect:
                raise RuntimeError(f"SOP mapping suspect: {mapping_stats.get('reason','')}")

            xdim, ydim, zdim = map(int, ct_img.GetSize())  # SITK is x,y,z
            mask = np.zeros((zdim, ydim, xdim), dtype=np.uint8)

            # Optional auto-detect 1-based VOC coords (conservative)
            if args.auto_one_based:
                applied_one_based = _detect_one_based(xml_files)

            # Collect all bboxes by slice index
            bboxes_by_slice: Dict[int, List[Tuple[int, int, int, int]]] = {}
            
            for xf in xml_files:
                sop_uid = _pick_sop_for_xml(xf, sop2z)
                if sop_uid is None:
                    missing_sop += 1
                    continue

                z = sop2z.get(sop_uid)
                if z is None or z < 0 or z >= zdim:
                    missing_sop += 1
                    continue

                bboxes = _parse_bboxes_from_xml(xf)
                if not bboxes:
                    continue

                for (xmin, ymin, xmax, ymax) in bboxes:
                    if applied_one_based:
                        xmin -= 1
                        ymin -= 1
                        xmax -= 1
                        ymax -= 1

                    # clamp
                    xmin = max(0, min(xdim - 1, xmin))
                    xmax = max(0, min(xdim - 1, xmax))
                    ymin = max(0, min(ydim - 1, ymin))
                    ymax = max(0, min(ydim - 1, ymax))
                    if xmax < xmin or ymax < ymin:
                        continue
                    
                    if args.use_medsam2:
                        shrunk = _shrink_bbox(
                            xmin, ymin, xmax, ymax,
                            shrink_frac=float(args.shrink_frac),
                            min_side_px=int(args.min_side_px),
                        )
                        if shrunk is None:
                            skipped_small += 1
                            continue
                        xmin, ymin, xmax, ymax = shrunk

                    if z not in bboxes_by_slice:
                        bboxes_by_slice[z] = []
                    bboxes_by_slice[z].append((xmin, ymin, xmax, ymax))
                    used += 1

            # Generate mask using either MedSAM2 or bbox filling
            if medsam2_segmenter is not None and bboxes_by_slice:
                # MedSAM2 mode: segment lesions from bboxes
                ct_array = sitk.GetArrayFromImage(ct_img)  # (z, y, x)
                mask = medsam2_segmenter.segment_multiple_bboxes(
                    ct_volume=ct_array,
                    bboxes_by_slice=bboxes_by_slice,
                    window_lower=float(args.window_lower),
                    window_upper=float(args.window_upper),
                )
                if bboxes_by_slice:
                    # Cropping to Z values of bboxes
                    zmin = min(bboxes_by_slice.keys())
                    zmax = max(bboxes_by_slice.keys())
                    margin = 2
                    lo = max(0, zmin - margin)
                    hi = min(mask.shape[0]-1, zmax + margin)
                    mask[:lo] = 0
                    mask[hi+1:] = 0
                    # Filter out air noise (voxels below -950 HU).
                    # Using -950 is conservative to avoid partial volume effects at lung boundaries,
                    # while effectively removing air artifacts inside the mask.
                    mask = mask.astype(bool)
                    mask &= (ct_array > -900)
                    mask = mask.astype(np.uint8)
            else:
                # Original bbox filling mode
                for z, slice_bboxes in bboxes_by_slice.items():
                    for (xmin, ymin, xmax, ymax) in slice_bboxes:
                        shrunk = _shrink_bbox(
                            xmin, ymin, xmax, ymax,
                            shrink_frac=float(args.shrink_frac),
                            min_side_px=int(args.min_side_px),
                        )
                        if shrunk is None:
                            skipped_small += 1
                            continue

                        xmin2, ymin2, xmax2, ymax2 = shrunk
                        mask[z, ymin2:ymax2 + 1, xmin2:xmax2 + 1] = 1

            # Optional: apply lung mask constraint
            if args.lung_mask_path:
                lung_mask_file = None
                if args.lung_mask_path == "auto":
                    # Try to read from manifest
                    m_temp = _load_json(manifest_path)
                    lung_path_str = m_temp.get("paths", {}).get("lung_mask", "")
                    if lung_path_str and Path(lung_path_str).exists():
                        lung_mask_file = Path(lung_path_str)
                else:
                    lung_mask_file = Path(args.lung_mask_path)
                
                if lung_mask_file and lung_mask_file.exists():
                    lung_img = sitk.ReadImage(str(lung_mask_file))
                    lung_array = sitk.GetArrayFromImage(lung_img)
                    # Intersection: tumor must be inside lung
                    mask = (mask.astype(np.uint8) & (lung_array > 0).astype(np.uint8))
                    print(f"Applied lung mask constraint from {lung_mask_file}")

            # Write mask
            seg_dir.mkdir(parents=True, exist_ok=True)
            out_img = sitk.GetImageFromArray(mask.astype(np.uint8))
            out_img.CopyInformation(ct_img)
            sitk.WriteImage(out_img, str(out_mask_path), useCompression=True)

            if used == 0:
                status = "ok_empty"
            elif mapping_stats.get("mapping_suspect", False):
                # Not a hard failure unless user requested, but we mark it clearly.
                status = "ok_mapping_suspect"

            runtime = round(time.time() - t0, 3)

            # Update case manifest (consistent schema)
            m = _load_json(manifest_path)
            m.setdefault("paths", {})
            m.setdefault("segmentation", {})

            m["paths"][manifest_key] = str(out_mask_path)
            m["segmentation"][manifest_key] = {
                "status": status,
                "runtime_sec": runtime,
                "segmentation_method": segmentation_method,
                "annotation_dir": str(ann_dir),
                "bbox_shrink_frac": float(args.shrink_frac) if segmentation_method == "bbox" else None,
                "bbox_min_side_px": int(args.min_side_px) if segmentation_method == "bbox" else None,
                "medsam2_window_lower": float(args.window_lower) if segmentation_method == "medsam2" else None,
                "medsam2_window_upper": float(args.window_upper) if segmentation_method == "medsam2" else None,
                "auto_one_based": bool(args.auto_one_based),
                "applied_one_based": bool(applied_one_based),
                "boxes_used": int(used),
                "missing_sop": int(missing_sop),
                "skipped_small_after_shrink": int(skipped_small),
                "ct_series_uid": ct_series_uid,
                "mapping_stats": mapping_stats,
            }
            _save_json(manifest_path, m)

        except Exception as e:
            status = "failed"
            err = repr(e)
            runtime = round(time.time() - t0, 3)

            # Best-effort manifest update
            try:
                m = _load_json(manifest_path)
                m.setdefault("segmentation", {})
                m["segmentation"]["tumor_bbox"] = {
                    "status": "failed",
                    "runtime_sec": runtime,
                    "error": err,
                    "bbox_shrink_frac": float(args.shrink_frac),
                    "bbox_min_side_px": int(args.min_side_px),
                    "auto_one_based": bool(args.auto_one_based),
                    "ct_series_uid": ct_series_uid,
                    "mapping_stats": mapping_stats,
                }
                _save_json(manifest_path, m)
            except Exception:
                pass

        rows.append({
            "patient_id": pid,
            "status": status,
            "tumor_mask_path": str(out_mask_path) if status in ("ok", "ok_empty", "ok_mapping_suspect") else "",
            "segmentation_method": segmentation_method,
            "boxes_used": int(used),
            "missing_sop": int(missing_sop),
            "skipped_small_after_shrink": int(skipped_small),
            "mapping_suspect": bool(mapping_stats.get("mapping_suspect", False)) if isinstance(mapping_stats, dict) else False,
            "error": err,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_index, index=False)
    print(f"Wrote tumor-mask summary: {out_index}")
    if "status" in out_df.columns:
        print(out_df["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
