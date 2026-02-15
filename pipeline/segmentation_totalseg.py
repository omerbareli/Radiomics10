# pipeline/segmentation_totalseg.py
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path

import SimpleITK as sitk

from pipeline.config import Config, ensure_project_dirs, ensure_case_dirs, get_case, verify_write_access

# TotalSegmentator lung lobes (reliable outputs in the "total" task)
LUNG_ROIS = [
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
]


def find_totalseg_exe() -> str:
    for name in ("TotalSegmentator", "totalsegmentator"):
        p = shutil.which(name)
        if p:
            return p
    raise RuntimeError(
        "TotalSegmentator CLI not found. Install with: python -m pip install totalsegmentator"
    )


def _load_manifest(case_dir: Path) -> dict:
    p = case_dir / "manifest.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _save_manifest(case_dir: Path, manifest: dict) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def _same_geom(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and a.GetSpacing() == b.GetSpacing()
        and a.GetOrigin() == b.GetOrigin()
        and a.GetDirection() == b.GetDirection()
    )


def union_masks(mask_paths: list[Path], out_path: Path, require_all: bool = True) -> None:
    """
    Union multiple binary ROI masks into one mask (uint8) preserving metadata.
    If require_all=True, raise if any ROI file is missing.
    """
    missing = [str(p) for p in mask_paths if not p.exists()]
    if missing and require_all:
        raise FileNotFoundError(f"Missing ROI masks: {missing}")

    imgs: list[sitk.Image] = []
    for p in mask_paths:
        if p.exists():
            imgs.append(sitk.ReadImage(str(p)))

    if not imgs:
        raise FileNotFoundError(f"No ROI masks found to union: {mask_paths}")

    ref = imgs[0]
    for img in imgs[1:]:
        if not _same_geom(ref, img):
            raise RuntimeError("ROI geometry mismatch (mask(s) do not match reference geometry)")

    union_arr = sitk.GetArrayFromImage(ref) > 0
    for img in imgs[1:]:
        union_arr |= (sitk.GetArrayFromImage(img) > 0)

    union = sitk.GetImageFromArray(union_arr.astype("uint8"))
    union.CopyInformation(ref)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(union, str(out_path))


def _select_case_ids(cfg: Config, limit: int, patients_csv: str) -> list[str]:
    # Prefer existing cases created by conversion
    case_ids = sorted([p.name for p in cfg.CASES_DIR.iterdir() if p.is_dir()])

    # Fallback: if no cases exist yet, derive from DICOM_ROOT
    if not case_ids:
        case_ids = sorted(
            [
                p.name
                for p in cfg.DICOM_ROOT.iterdir()
                if p.is_dir() and p.name.startswith("Lung_Dx-")
            ]
        )

    if patients_csv.strip():
        wanted = {x.strip() for x in patients_csv.split(",") if x.strip()}
        case_ids = [cid for cid in case_ids if cid in wanted]

    if limit and limit > 0:
        case_ids = case_ids[:limit]

    return case_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Process only first N patients (0 = all).")
    parser.add_argument("--patients", type=str, default="", help="Comma-separated patient IDs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing lung_mask.nii.gz and ROI dir.")
    parser.add_argument("--fast", action="store_true", help="Use TotalSegmentator --fast.")
    parser.add_argument("--require-all-rois", action="store_true", help="Fail if any ROI mask is missing.")
    args = parser.parse_args()

    cfg = Config()
    ensure_project_dirs(cfg)

    # Pre-flight permission check on mounted data directory
    verify_write_access(cfg.DATA_ROOT)

    exe = find_totalseg_exe()

    case_ids = _select_case_ids(cfg, args.limit, args.patients)
    if not case_ids:
        raise RuntimeError(
            "No cases found to segment (no folders under data/cases and no Lung_Dx-* under DICOM_ROOT)."
        )

    # Global summary CSV
    log_csv = cfg.DATA_DIR / "index_segmentation.csv"
    write_header = not log_csv.exists()

    with open(log_csv, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient_id",
                "ct_path",
                "totalseg_dir",
                "lung_mask_path",
                "status",
                "runtime_sec",
                "error",
            ],
        )
        if write_header:
            writer.writeheader()

        for pid in case_ids:
            case = get_case(cfg, pid)
            ensure_case_dirs(case)

            # Best-effort case_dir for manifest update
            case_dir = getattr(case, "case_dir", None)
            if case_dir is None:
                case_dir = cfg.CASES_DIR / pid

            ct_path = case.nifti_dir / "ct.nii.gz"
            if not ct_path.exists():
                writer.writerow(
                    {
                        "patient_id": pid,
                        "ct_path": str(ct_path),
                        "totalseg_dir": "",
                        "lung_mask_path": "",
                        "status": "missing_ct",
                        "runtime_sec": 0.0,
                        "error": "ct.nii.gz not found in case/nifti",
                    }
                )
                # Update manifest (optional but consistent)
                m = _load_manifest(case_dir)
                m.setdefault("segmentation", {})
                m["segmentation"]["totalseg"] = {
                    "status": "missing_ct",
                    "runtime_sec": 0.0,
                    "fast": bool(args.fast),
                    "rois": LUNG_ROIS,
                    "error": "ct.nii.gz not found in case/nifti",
                }
                _save_manifest(case_dir, m)
                continue

            out_dir = case.seg_dir / "totalseg_roi"
            lung_mask_path = case.seg_dir / "lung_mask.nii.gz"

            if lung_mask_path.exists() and not args.overwrite:
                writer.writerow(
                    {
                        "patient_id": pid,
                        "ct_path": str(ct_path),
                        "totalseg_dir": str(out_dir),
                        "lung_mask_path": str(lung_mask_path),
                        "status": "skipped_exists",
                        "runtime_sec": 0.0,
                        "error": "",
                    }
                )
                # Keep manifest consistent
                m = _load_manifest(case_dir)
                m.setdefault("paths", {})
                m.setdefault("segmentation", {})
                m["paths"]["lung_mask"] = str(lung_mask_path)
                m["paths"]["totalseg_dir"] = str(out_dir)
                m["segmentation"]["totalseg"] = {
                    "status": "skipped_exists",
                    "runtime_sec": 0.0,
                    "fast": bool(args.fast),
                    "rois": LUNG_ROIS,
                    "error": "",
                }
                _save_manifest(case_dir, m)
                continue

            t0 = time.time()
            err = ""
            status = "ok"

            try:
                # Avoid stale ROI files when overwriting
                if args.overwrite and out_dir.exists():
                    shutil.rmtree(out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    exe,
                    "-i",
                    str(ct_path),
                    "-o",
                    str(out_dir),
                    "--task",
                    "total",
                    "--roi_subset",
                    *LUNG_ROIS,
                ]
                if args.fast:
                    cmd += ["--fast"]

                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode != 0:
                    tail = (res.stderr or "")[-4000:]
                    raise RuntimeError(
                        f"TotalSegmentator failed (rc={res.returncode}). stderr tail:\n{tail}"
                    )

                roi_paths = [out_dir / f"{roi}.nii.gz" for roi in LUNG_ROIS]
                union_masks(roi_paths, lung_mask_path, require_all=args.require_all_rois)

            except Exception as e:
                status = "failed"
                err = repr(e)

            runtime = round(time.time() - t0, 3)

            # Update manifest
            m = _load_manifest(case_dir)
            m.setdefault("paths", {})
            m.setdefault("segmentation", {})

            # Only write lung_mask path if success
            if status == "ok":
                m["paths"]["lung_mask"] = str(lung_mask_path)
            m["paths"]["totalseg_dir"] = str(out_dir)

            m["segmentation"]["totalseg"] = {
                "status": status,
                "runtime_sec": runtime,
                "fast": bool(args.fast),
                "rois": LUNG_ROIS,
                "error": err,
            }
            _save_manifest(case_dir, m)

            writer.writerow(
                {
                    "patient_id": pid,
                    "ct_path": str(ct_path),
                    "totalseg_dir": str(out_dir),
                    "lung_mask_path": str(lung_mask_path) if status == "ok" else "",
                    "status": status,
                    "runtime_sec": runtime,
                    "error": err,
                }
            )

    print(f"Wrote segmentation summary: {log_csv}")


if __name__ == "__main__":
    main()
