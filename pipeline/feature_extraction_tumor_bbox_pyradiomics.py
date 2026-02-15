# pipeline/feature_extraction_tumor_bbox_pyradiomics.py
from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

from pipeline.config import Config, ensure_project_dirs, ensure_case_dirs, get_case, verify_write_access


# ----------------------------
# Geometry + resampling utils
# ----------------------------

def _same_geom_tol(a: sitk.Image, b: sitk.Image, tol: float = 1e-5) -> bool:
    """Geometry equality with float tolerance."""
    if a.GetSize() != b.GetSize():
        return False
    asp = np.array(a.GetSpacing(), dtype=float)
    bsp = np.array(b.GetSpacing(), dtype=float)
    ao = np.array(a.GetOrigin(), dtype=float)
    bo = np.array(b.GetOrigin(), dtype=float)
    ad = np.array(a.GetDirection(), dtype=float)
    bd = np.array(b.GetDirection(), dtype=float)
    return (
        np.allclose(asp, bsp, atol=tol, rtol=0.0)
        and np.allclose(ao, bo, atol=tol, rtol=0.0)
        and np.allclose(ad, bd, atol=tol, rtol=0.0)
    )


def resample_to_reference(
    moving: sitk.Image,
    reference: sitk.Image,
    interpolator=sitk.sitkLinear,
    default_value: float = 0.0,
    out_pixel: int = sitk.sitkFloat32,
) -> sitk.Image:
    """
    Resample `moving` onto `reference` grid (size/spacing/origin/direction).

    IMPORTANT: default out_pixel is Float32 for safety (especially for PET).
    """
    return sitk.Resample(
        moving,
        reference,
        sitk.Transform(),
        interpolator,
        float(default_value),
        out_pixel,
    )


def _ensure_binary_mask(mask: sitk.Image) -> sitk.Image:
    """Ensure mask is uint8 {0,1}."""
    return sitk.Cast(mask > 0, sitk.sitkUInt8)


def _align_mask_to_ct(mask: sitk.Image, ct: sitk.Image, tol: float = 1e-5) -> Tuple[sitk.Image, bool]:
    """
    Ensure mask is on CT geometry. If mismatched, resample (NN) and binarize.
    Returns (mask_aligned, was_resampled).
    """
    if _same_geom_tol(mask, ct, tol=tol):
        return _ensure_binary_mask(mask), False
    mask_rs = resample_to_reference(mask, ct, interpolator=sitk.sitkNearestNeighbor, out_pixel=sitk.sitkUInt8)
    return _ensure_binary_mask(mask_rs), True


def _count_mask_voxels(mask: sitk.Image) -> int:
    """Count voxels > 0 using a view when possible."""
    try:
        arr = sitk.GetArrayViewFromImage(mask)
    except Exception:
        arr = sitk.GetArrayFromImage(mask)
    return int((arr > 0).sum())


def _mask_physical_volume_mm3(mask_voxels: int, ct: sitk.Image) -> float:
    sp = np.array(ct.GetSpacing(), dtype=float)  # (x,y,z) in mm
    return float(mask_voxels) * float(np.prod(sp))


def _extract_masked_values(image: sitk.Image, mask: sitk.Image) -> np.ndarray:
    """
    Extract voxel values inside mask (label==1) as a 1D numpy array.
    Only copies the masked values (ROI), not the entire volume.
    """
    img_arr = sitk.GetArrayViewFromImage(image)   # z,y,x
    m_arr = sitk.GetArrayViewFromImage(mask)      # z,y,x
    vals = np.asarray(img_arr[m_arr > 0], dtype=np.float32)
    return vals


def _ct_mask_sanity(vals_hu: np.ndarray) -> Dict[str, Any]:
    """
    Basic 'is this ROI plausibly inside patient' sanity for CT (HU).

    Goal: catch obvious misplacement (outside volume / pure air).
    We keep this conservative to avoid false negatives on lung regions.

    Rules (conservative):
      - If ROI is empty -> fail.
      - If max HU is very low (e.g., < -850), it's basically all air -> fail.
      - If mean < -950 AND max < -500 -> fail (extremely air-like).
      - Otherwise pass but report metrics.
    """
    out: Dict[str, Any] = {}
    if vals_hu.size == 0:
        out["ok"] = False
        out["reason"] = "empty_mask_after_alignment"
        return out

    vmin = float(np.min(vals_hu))
    vmax = float(np.max(vals_hu))
    vmean = float(np.mean(vals_hu))
    vstd = float(np.std(vals_hu))
    frac_over_m500 = float(np.mean(vals_hu > -500.0))  # tissue-ish fraction
    frac_over_0 = float(np.mean(vals_hu > 0.0))        # soft tissue/bone-ish fraction

    out.update(
        {
            "ok": True,
            "min_hu": vmin,
            "max_hu": vmax,
            "mean_hu": vmean,
            "std_hu": vstd,
            "frac_over_-500": frac_over_m500,
            "frac_over_0": frac_over_0,
        }
    )

    # Hard fails: almost certainly wrong place
    if vmax < -850.0:
        out["ok"] = False
        out["reason"] = "ct_roi_all_air_max<-850"
        return out

    if (vmean < -950.0) and (vmax < -500.0):
        out["ok"] = False
        out["reason"] = "ct_roi_extreme_air_mean<-950_and_max<-500"
        return out

    # Soft warning (doesn't fail): very air-dominant
    if (vmean < -850.0) and (frac_over_m500 < 0.001):
        out["warning"] = "ct_roi_very_air_dominant"

    return out


def _diagnostics_subset(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep a small useful subset of PyRadiomics diagnostics for QC/debug.
    (Avoid dumping huge diagnostics noise.)
    """
    keep_prefixes = (
        "diagnostics_Versions",
        "diagnostics_Configuration",
        "diagnostics_Image-original",
        "diagnostics_Mask-original",
    )
    out: Dict[str, Any] = {}
    for k, v in result.items():
        if not k.startswith("diagnostics"):
            continue
        if k.startswith(keep_prefixes):
            out[k] = v
    return out


# ----------------------------
# Manifest helpers
# ----------------------------

def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str))


def _select_case_ids(cfg: Config, limit: int, patients_csv: str) -> list[str]:
    case_ids = sorted([p.name for p in cfg.CASES_DIR.iterdir() if p.is_dir()])

    if patients_csv.strip():
        wanted = {x.strip() for x in patients_csv.split(",") if x.strip()}
        case_ids = [cid for cid in case_ids if cid in wanted]

    if limit and limit > 0:
        case_ids = case_ids[:limit]

    return case_ids


def _resolve_mask_path_from_manifest(manifest: dict, seg_dir: Path) -> Tuple[Path | None, str]:
    """
    Resolve mask path from manifest, prioritizing MedSAM2 (tumor_mask_fm) over bbox.
    
    Returns:
        (mask_path, source_key) where source_key is the manifest key used.
        Returns (None, "") if no mask is found.
    """
    paths = manifest.get("paths", {})
    
    # Priority order: MedSAM2 first, then bbox
    priority_keys = ["tumor_mask_fm", "tumor_bbox_mask"]
    
    for key in priority_keys:
        path_str = paths.get(key, "")
        if path_str:
            p = Path(path_str)
            if p.exists() and p.is_file():
                return p, key
    
    # Fallback: check if files exist in seg_dir with known names
    fallback_names = [
        ("tumor_mask_fm.nii.gz", "tumor_mask_fm"),
        ("tumor_bbox_mask.nii.gz", "tumor_bbox_mask"),
    ]
    for fname, key in fallback_names:
        p = seg_dir / fname
        if p.exists() and p.is_file():
            return p, key
    
    return None, ""


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    # Allow split modality params (recommended); fallback uses --params for both.
    ap.add_argument("--params", type=str, default="pipeline/radiomics_ct_pet.yaml",
                    help="Default PyRadiomics YAML params file (used for both CT/PET unless overridden).")
    ap.add_argument("--params-ct", type=str, default="", help="Optional CT-specific PyRadiomics YAML.")
    ap.add_argument("--params-pet", type=str, default="", help="Optional PET-specific PyRadiomics YAML.")

    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--patients", type=str, default="")
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--save-resampled-pet", action="store_true",
                    help="Save tumor_bbox_pet_resampled_to_ct.nii.gz for debugging/repro.")

    ap.add_argument("--min-mask-voxels", type=int, default=50,
                    help="Skip feature extraction if mask has fewer than this many voxels.")
    ap.add_argument("--min-mask-volume-mm3", type=float, default=0.0,
                    help="Optional physical-volume gate (mm^3). 0 disables.")

    ap.add_argument("--geom-tol", type=float, default=1e-5,
                    help="Tolerance for geometry comparisons (origin/spacing/direction).")
    ap.add_argument("--fail-on-mask-mismatch", action="store_true",
                    help="Fail instead of resampling if mask geometry doesn't match CT.")
    ap.add_argument("--fail-on-ct-mask-sanity", action="store_true",
                    help="Fail case if CT sanity check says ROI is implausible (safer).")

    # Flexible mask selection
    ap.add_argument("--mask-name", type=str, default="tumor_bbox_mask.nii.gz",
                    help="Mask filename to look for in seg/ directory. "
                         "Use 'tumor_mask_fm.nii.gz' for MedSAM2 outputs.")
    ap.add_argument("--use-manifest-paths", action="store_true",
                    help="Read mask path from manifest.json instead of hardcoded filename. "
                         "Prioritizes tumor_mask_fm (MedSAM2) over tumor_bbox_mask.")

    args = ap.parse_args()

    cfg = Config()
    ensure_project_dirs(cfg)

    # Pre-flight permission check on mounted data directory
    verify_write_access(cfg.DATA_ROOT)

    params_default = Path(args.params)
    if not params_default.exists():
        raise FileNotFoundError(f"PyRadiomics params YAML not found: {params_default}")

    params_ct = Path(args.params_ct) if args.params_ct.strip() else params_default
    params_pet = Path(args.params_pet) if args.params_pet.strip() else params_default

    if not params_ct.exists():
        raise FileNotFoundError(f"CT params YAML not found: {params_ct}")
    if not params_pet.exists():
        raise FileNotFoundError(f"PET params YAML not found: {params_pet}")

    extractor_ct = featureextractor.RadiomicsFeatureExtractor(str(params_ct))
    extractor_pet = featureextractor.RadiomicsFeatureExtractor(str(params_pet))

    case_ids = _select_case_ids(cfg, args.limit, args.patients)
    if not case_ids:
        raise RuntimeError("No cases found under data/cases. Run conversion + XML->mask first.")

    rows = []

    for pid in case_ids:
        case = get_case(cfg, pid)
        ensure_case_dirs(case)

        features_dir = case.features_dir

        ct_path = case.nifti_dir / "ct.nii.gz"
        pet_path = case.nifti_dir / "pet.nii.gz"

        out_ct_json = features_dir / "tumor_bbox_ct_features.json"
        out_pet_json = features_dir / "tumor_bbox_pet_features.json"
        out_pet_resampled = features_dir / "tumor_bbox_pet_resampled_to_ct.nii.gz"

        status_ct = "not_run"
        status_pet = "not_run"
        err_ct = ""
        err_pet = ""

        manifest = _load_manifest(case.manifest_path)
        manifest.setdefault("case_id", pid)  # don't fight existing conventions
        manifest.setdefault("paths", {})
        manifest.setdefault("segmentation", {})
        manifest.setdefault("features", {})
        manifest["paths"].setdefault("features_dir", str(features_dir))

        # ----------------
        # Resolve mask path (flexible: manifest or CLI argument)
        # ----------------
        mask_source_key = ""
        if args.use_manifest_paths:
            mask_path, mask_source_key = _resolve_mask_path_from_manifest(manifest, case.seg_dir)
        else:
            mask_path = case.seg_dir / args.mask_name
            mask_source_key = Path(args.mask_name).stem  # e.g., "tumor_mask_fm" or "tumor_bbox_mask"

        feat_node = manifest["features"].setdefault("tumor_bbox_radiomics", {})
        feat_node["updated_at"] = datetime.utcnow().isoformat() + "Z"
        feat_node["params_ct_yaml"] = str(params_ct)
        feat_node["params_pet_yaml"] = str(params_pet)
        feat_node["min_mask_voxels"] = int(args.min_mask_voxels)
        feat_node["min_mask_volume_mm3"] = float(args.min_mask_volume_mm3)
        feat_node["geom_tol"] = float(args.geom_tol)
        feat_node["fail_on_ct_mask_sanity"] = bool(args.fail_on_ct_mask_sanity)
        feat_node["source_mask"] = str(mask_path) if mask_path and mask_path.exists() else ""
        feat_node["source_mask_key"] = mask_source_key

        # ----------------
        # Prereqs
        # ----------------
        if not ct_path.exists():
            status_ct = "missing_ct"
            status_pet = "missing_ct"
            feat_node["status_ct"] = status_ct
            feat_node["status_pet"] = status_pet
            feat_node["overall"] = "failed"
            _save_manifest(case.manifest_path, manifest)
            rows.append({
                "patient_id": pid,
                "overall": "failed",
                "status_ct": status_ct,
                "status_pet": status_pet,
                "tumor_bbox_ct_features": "",
                "tumor_bbox_pet_features": "",
                "error_ct": "",
                "error_pet": "",
            })
            continue

        if mask_path is None or not mask_path.exists() or not mask_path.is_file():
            status_ct = "missing_tumor_mask"
            status_pet = "missing_tumor_mask"
            feat_node["status_ct"] = status_ct
            feat_node["status_pet"] = status_pet
            feat_node["overall"] = "no_tumor_bbox"
            feat_node["mask_search_mode"] = "manifest" if args.use_manifest_paths else f"filename:{args.mask_name}"
            _save_manifest(case.manifest_path, manifest)
            rows.append({
                "patient_id": pid,
                "overall": "no_tumor_bbox",
                "status_ct": status_ct,
                "status_pet": status_pet,
                "source_mask": "",
                "tumor_bbox_ct_features": "",
                "tumor_bbox_pet_features": "",
                "error_ct": "",
                "error_pet": "",
            })
            continue

        # Read CT + mask once
        ct = sitk.ReadImage(str(ct_path))
        mask_raw = sitk.ReadImage(str(mask_path))

        # Align mask to CT grid (robustness critical)
        if not _same_geom_tol(mask_raw, ct, tol=args.geom_tol) and args.fail_on_mask_mismatch:
            status_ct = "failed_mask_geom_mismatch"
            status_pet = "failed_mask_geom_mismatch"
            err = "Mask geometry does not match CT (and --fail-on-mask-mismatch was set)."
            feat_node["mask_geom_mismatch"] = True
            feat_node["status_ct"] = status_ct
            feat_node["status_pet"] = status_pet
            feat_node["overall"] = "failed"
            feat_node["ct_error"] = err
            feat_node["pet_error"] = err
            _save_manifest(case.manifest_path, manifest)
            rows.append({
                "patient_id": pid,
                "overall": "failed",
                "status_ct": status_ct,
                "status_pet": status_pet,
                "tumor_bbox_ct_features": "",
                "tumor_bbox_pet_features": "",
                "error_ct": err,
                "error_pet": err,
            })
            continue

        mask, mask_resampled = _align_mask_to_ct(mask_raw, ct, tol=args.geom_tol)
        feat_node["mask_resampled_to_ct"] = bool(mask_resampled)

        vox = _count_mask_voxels(mask)
        feat_node["tumor_bbox_mask_voxels"] = int(vox)

        vol_mm3 = _mask_physical_volume_mm3(vox, ct)
        feat_node["tumor_bbox_mask_volume_mm3"] = float(vol_mm3)

        # Gates
        if vox < int(args.min_mask_voxels):
            status_ct = "skipped_small_mask"
            status_pet = "skipped_small_mask"
            feat_node["status_ct"] = status_ct
            feat_node["status_pet"] = status_pet
            feat_node["overall"] = "no_tumor_bbox"
            _save_manifest(case.manifest_path, manifest)
            rows.append({
                "patient_id": pid,
                "overall": "no_tumor_bbox",
                "status_ct": status_ct,
                "status_pet": status_pet,
                "tumor_bbox_ct_features": "",
                "tumor_bbox_pet_features": "",
                "mask_voxels": int(vox),
                "mask_volume_mm3": float(vol_mm3),
                "mask_resampled_to_ct": bool(mask_resampled),
                "error_ct": "",
                "error_pet": "",
            })
            continue

        if float(args.min_mask_volume_mm3) > 0.0 and vol_mm3 < float(args.min_mask_volume_mm3):
            status_ct = "skipped_small_mask_volume"
            status_pet = "skipped_small_mask_volume"
            feat_node["status_ct"] = status_ct
            feat_node["status_pet"] = status_pet
            feat_node["overall"] = "no_tumor_bbox"
            _save_manifest(case.manifest_path, manifest)
            rows.append({
                "patient_id": pid,
                "overall": "no_tumor_bbox",
                "status_ct": status_ct,
                "status_pet": status_pet,
                "tumor_bbox_ct_features": "",
                "tumor_bbox_pet_features": "",
                "mask_voxels": int(vox),
                "mask_volume_mm3": float(vol_mm3),
                "mask_resampled_to_ct": bool(mask_resampled),
                "error_ct": "",
                "error_pet": "",
            })
            continue

        # CT sanity check (catch misregistered mask that still has voxels)
        ct_vals = _extract_masked_values(ct, mask)
        sanity = _ct_mask_sanity(ct_vals)
        feat_node["ct_mask_sanity"] = sanity

        if not sanity.get("ok", True):
            status_ct = "failed_mask_sanity"
            status_pet = "failed_mask_sanity"
            err = f"CT mask sanity failed: {sanity.get('reason', 'unknown')}"
            feat_node["status_ct"] = status_ct
            feat_node["status_pet"] = status_pet
            feat_node["overall"] = "failed" if args.fail_on_ct_mask_sanity else "no_tumor_bbox"
            feat_node["ct_error"] = err
            feat_node["pet_error"] = err

            # If user didn't request hard fail, treat as "no usable bbox"
            if not args.fail_on_ct_mask_sanity:
                status_ct = "skipped_mask_sanity"
                status_pet = "skipped_mask_sanity"
                feat_node["status_ct"] = status_ct
                feat_node["status_pet"] = status_pet
                feat_node["overall"] = "no_tumor_bbox"

            _save_manifest(case.manifest_path, manifest)
            rows.append({
                "patient_id": pid,
                "overall": feat_node["overall"],
                "status_ct": status_ct,
                "status_pet": status_pet,
                "tumor_bbox_ct_features": "",
                "tumor_bbox_pet_features": "",
                "mask_voxels": int(vox),
                "mask_volume_mm3": float(vol_mm3),
                "mask_resampled_to_ct": bool(mask_resampled),
                "error_ct": err,
                "error_pet": err,
            })
            continue

        # ----------------
        # CT features
        # ----------------
        if out_ct_json.exists() and (not args.overwrite):
            status_ct = "skipped_exists"
            manifest["paths"]["tumor_bbox_ct_features"] = str(out_ct_json)
        else:
            try:
                ct_result = extractor_ct.execute(ct, mask)
                ct_features = {k: v for k, v in ct_result.items() if not k.startswith("diagnostics")}
                # Add source mask metadata
                ct_features["_source_mask"] = str(mask_path)
                ct_features["_source_mask_key"] = mask_source_key
                out_ct_json.write_text(json.dumps(ct_features, indent=2, default=str))
                status_ct = "ok"
                manifest["paths"]["tumor_bbox_ct_features"] = str(out_ct_json)

                feat_node["ct_diagnostics_subset"] = _diagnostics_subset(ct_result)
            except Exception:
                status_ct = "failed"
                err_ct = traceback.format_exc(limit=20)

        # ----------------
        # PET features
        # ----------------
        if not pet_path.exists():
            status_pet = "missing_pet"
        else:
            if out_pet_json.exists() and (not args.overwrite):
                status_pet = "skipped_exists"
                manifest["paths"]["tumor_bbox_pet_features"] = str(out_pet_json)
            else:
                try:
                    pet = sitk.ReadImage(str(pet_path))

                    # Resample PET -> CT grid if needed (FORCE Float32 output)
                    if _same_geom_tol(pet, ct, tol=args.geom_tol):
                        pet_on_ct = sitk.Cast(pet, sitk.sitkFloat32)
                        pet_resampled = False
                    else:
                        pet_on_ct = resample_to_reference(
                            pet,
                            ct,
                            interpolator=sitk.sitkLinear,
                            default_value=0.0,
                            out_pixel=sitk.sitkFloat32,
                        )
                        pet_resampled = True

                    feat_node["pet_resampled_to_ct"] = bool(pet_resampled)

                    if args.save_resampled_pet:
                        sitk.WriteImage(pet_on_ct, str(out_pet_resampled), useCompression=True)
                        manifest["paths"]["tumor_bbox_pet_resampled_to_ct"] = str(out_pet_resampled)

                    pet_result = extractor_pet.execute(pet_on_ct, mask)
                    pet_features = {k: v for k, v in pet_result.items() if not k.startswith("diagnostics")}
                    # Add source mask metadata
                    pet_features["_source_mask"] = str(mask_path)
                    pet_features["_source_mask_key"] = mask_source_key
                    out_pet_json.write_text(json.dumps(pet_features, indent=2, default=str))
                    status_pet = "ok"
                    manifest["paths"]["tumor_bbox_pet_features"] = str(out_pet_json)

                    feat_node["pet_diagnostics_subset"] = _diagnostics_subset(pet_result)

                except Exception:
                    status_pet = "failed"
                    err_pet = traceback.format_exc(limit=20)

        # -------------
        # Overall status
        # -------------
        if status_ct in ("ok", "skipped_exists"):
            if status_pet in ("ok", "skipped_exists"):
                overall = "ok_ct_pet"
            elif status_pet == "missing_pet":
                overall = "ok_ct_only"
            elif status_pet == "failed":
                overall = "partial_failed_pet"
            else:
                overall = "ok_ct_only"
        elif status_ct.startswith("skipped_"):
            overall = "no_tumor_bbox"
        else:
            overall = "failed"

        feat_node["status_ct"] = status_ct
        feat_node["status_pet"] = status_pet
        feat_node["overall"] = overall
        if err_ct:
            feat_node["ct_error"] = err_ct
        if err_pet:
            feat_node["pet_error"] = err_pet

        manifest["updated_at"] = datetime.utcnow().isoformat() + "Z"
        _save_manifest(case.manifest_path, manifest)

        rows.append({
            "patient_id": pid,
            "overall": overall,
            "status_ct": status_ct,
            "status_pet": status_pet,
            "source_mask": str(mask_path),
            "source_mask_key": mask_source_key,
            "tumor_bbox_ct_features": str(out_ct_json) if out_ct_json.exists() else "",
            "tumor_bbox_pet_features": str(out_pet_json) if out_pet_json.exists() else "",
            "mask_voxels": int(vox),
            "mask_volume_mm3": float(vol_mm3),
            "mask_resampled_to_ct": bool(mask_resampled),
            "error_ct": err_ct,
            "error_pet": err_pet,
        })

    df = pd.DataFrame(rows)
    out_index = cfg.DATA_DIR / "index_tumor_bbox_features.csv"
    out_index.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_index, index=False)
    print(f"Wrote tumor-bbox features summary: {out_index}")
    if "overall" in df.columns:
        print(df["overall"].value_counts().to_string())


if __name__ == "__main__":
    main()
