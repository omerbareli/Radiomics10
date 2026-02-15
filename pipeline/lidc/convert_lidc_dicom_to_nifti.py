# pipeline/lidc/convert_lidc_dicom_to_nifti.py
"""Convert LIDC-IDRI DICOM CT series to NIfTI with SOP_UID extraction.

This script:
1. Finds CT series (filters out DX/radiograph series)
2. Extracts SOPInstanceUID per slice for annotation matching
3. Converts to NIfTI with proper orientation
4. Saves manifest with SOP→slice mapping

Usage:
    python -m pipeline.lidc.convert_lidc_dicom_to_nifti --patient LIDC-IDRI-0001
    python -m pipeline.lidc.convert_lidc_dicom_to_nifti --all --limit 10
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pydicom
import SimpleITK as sitk
from tqdm import tqdm

from pipeline.lidc.config import (
    LIDCConfig,
    ensure_case_dirs,
    ensure_lidc_dirs,
    get_lidc_case,
    list_available_patients,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class SliceInfo:
    """Information about a single DICOM slice."""
    file_path: Path
    sop_instance_uid: str
    z_position: float
    slice_index: int  # Index in sorted order


@dataclass
class SeriesInfo:
    """Information about a CT series."""
    series_uid: str
    study_uid: str
    modality: str
    series_description: str
    num_slices: int
    slice_thickness: Optional[float]
    pixel_spacing: Optional[Tuple[float, float]]
    slices: List[SliceInfo]


def _read_dicom_header(dcm_path: Path) -> Dict[str, Any]:
    """Read key DICOM header fields."""
    try:
        dcm = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
    except Exception as e:
        logger.warning(f"Failed to read DICOM: {dcm_path.name}: {e}")
        return {}
    
    def get_val(attr: str, default: Any = None) -> Any:
        return getattr(dcm, attr, default)
    
    # Get ImagePositionPatient[2] for Z position
    ipp = get_val("ImagePositionPatient", [0, 0, 0])
    z_position = float(ipp[2]) if len(ipp) >= 3 else 0.0
    
    # Get pixel spacing
    ps = get_val("PixelSpacing", None)
    pixel_spacing = (float(ps[0]), float(ps[1])) if ps else None
    
    return {
        "sop_instance_uid": str(get_val("SOPInstanceUID", "")),
        "series_uid": str(get_val("SeriesInstanceUID", "")),
        "study_uid": str(get_val("StudyInstanceUID", "")),
        "modality": str(get_val("Modality", "")),
        "series_description": str(get_val("SeriesDescription", "")),
        "slice_thickness": float(get_val("SliceThickness", 0)) or None,
        "pixel_spacing": pixel_spacing,
        "z_position": z_position,
        "file_path": dcm_path,
    }


def find_dicom_files(patient_dir: Path) -> List[Path]:
    """Find all DICOM files in patient directory tree."""
    dcm_files = []
    for ext in ["*.dcm", "*.DCM"]:
        dcm_files.extend(patient_dir.rglob(ext))
    
    # Also check for extensionless DICOM files (common in TCIA)
    for f in patient_dir.rglob("*"):
        if f.is_file() and f.suffix == "" and not f.name.startswith("."):
            # Quick check if it's DICOM
            try:
                with open(f, "rb") as fh:
                    fh.seek(128)
                    if fh.read(4) == b"DICM":
                        dcm_files.append(f)
            except Exception:
                pass
    
    return sorted(set(dcm_files))


def group_by_series(dcm_files: List[Path]) -> Dict[str, List[Dict[str, Any]]]:
    """Group DICOM files by SeriesInstanceUID."""
    series_map: Dict[str, List[Dict[str, Any]]] = {}
    
    for dcm_file in dcm_files:
        header = _read_dicom_header(dcm_file)
        if not header or not header.get("series_uid"):
            continue
        
        series_uid = header["series_uid"]
        if series_uid not in series_map:
            series_map[series_uid] = []
        series_map[series_uid].append(header)
    
    return series_map


def select_ct_series(
    series_map: Dict[str, List[Dict[str, Any]]],
    patient_id: str = "",
) -> Tuple[Optional[SeriesInfo], Dict[str, Any]]:
    """Select the best CT series from available series with DETERMINISTIC logging.
    
    Selection criteria (in priority order):
    1. Must be CT modality (not DX, CR, etc.)
    2. More slices is better
    3. Thinner slice thickness is better (tiebreaker)
    
    Returns:
        Tuple of (selected SeriesInfo or None, selection_log dict)
    """
    ct_candidates = []
    rejected_series = []
    
    for series_uid, slices in series_map.items():
        if not slices:
            continue
        
        first_slice = slices[0]
        modality = first_slice.get("modality", "").upper()
        
        # Track non-CT series
        if modality != "CT":
            rejected_series.append({
                "series_uid": series_uid[:20] + "...",
                "modality": modality,
                "num_slices": len(slices),
                "reason": f"Non-CT modality: {modality}",
            })
            continue
        
        # Sort slices by Z position
        sorted_slices = sorted(slices, key=lambda s: s["z_position"])
        
        # Build SliceInfo list
        slice_infos = []
        for idx, s in enumerate(sorted_slices):
            slice_infos.append(SliceInfo(
                file_path=s["file_path"],
                sop_instance_uid=s["sop_instance_uid"],
                z_position=s["z_position"],
                slice_index=idx,
            ))
        
        series_info = SeriesInfo(
            series_uid=series_uid,
            study_uid=first_slice.get("study_uid", ""),
            modality=modality,
            series_description=first_slice.get("series_description", ""),
            num_slices=len(slice_infos),
            slice_thickness=first_slice.get("slice_thickness"),
            pixel_spacing=first_slice.get("pixel_spacing"),
            slices=slice_infos,
        )
        
        ct_candidates.append(series_info)
    
    selection_log = {
        "total_series": len(series_map),
        "ct_candidates": len(ct_candidates),
        "rejected_series": rejected_series,
        "selection_reason": None,
    }
    
    if not ct_candidates:
        selection_log["selection_reason"] = "No CT series found"
        return None, selection_log
    
    # Score and select best
    def score(s: SeriesInfo) -> Tuple[int, int]:
        # Primary: more slices is better
        # Secondary: prefer thinner slices (inverted for sorting)
        thickness_score = 0 if s.slice_thickness is None else -int(s.slice_thickness * 100)
        return (s.num_slices, thickness_score)
    
    ct_candidates.sort(key=score, reverse=True)
    
    # Log selection decision
    selected = ct_candidates[0]
    runner_up = ct_candidates[1] if len(ct_candidates) > 1 else None
    
    if len(ct_candidates) == 1:
        reason = f"Only CT series available (slices={selected.num_slices}, thickness={selected.slice_thickness}mm)"
    elif runner_up:
        if selected.num_slices > runner_up.num_slices:
            reason = (
                f"Selected series with MAX slices: {selected.num_slices} slices "
                f"(runner-up: {runner_up.num_slices} slices)"
            )
        elif selected.num_slices == runner_up.num_slices:
            reason = (
                f"Tiebreaker: thinner slices ({selected.slice_thickness}mm vs {runner_up.slice_thickness}mm), "
                f"both have {selected.num_slices} slices"
            )
        else:
            reason = f"Best score: slices={selected.num_slices}, thickness={selected.slice_thickness}mm"
    else:
        reason = f"Best candidate: slices={selected.num_slices}, thickness={selected.slice_thickness}mm"
    
    selection_log["selection_reason"] = reason
    selection_log["selected_series_uid"] = selected.series_uid
    selection_log["candidates"] = [
        {
            "series_uid": c.series_uid[:30] + "...",
            "slices": c.num_slices,
            "thickness_mm": c.slice_thickness,
            "description": c.series_description[:50],
        }
        for c in ct_candidates[:5]  # Top 5 candidates
    ]
    
    # Log the decision
    if patient_id:
        logger.info(f"[{patient_id}] Series selection: {reason}")
    
    return selected, selection_log


def convert_series_to_nifti(series: SeriesInfo, output_path: Path) -> sitk.Image:
    """Convert DICOM series to NIfTI using SimpleITK.
    
    Args:
        series: SeriesInfo with slice file paths
        output_path: Where to save the NIfTI file
        
    Returns:
        SimpleITK Image object
    """
    # Get sorted file paths
    file_paths = [str(s.file_path) for s in series.slices]
    
    # Read with SimpleITK's GDCM reader
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_paths)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    
    try:
        image = reader.Execute()
    except Exception as e:
        logger.error(f"SimpleITK read failed: {e}")
        raise
    
    # Ensure 3D
    if image.GetDimension() == 2:
        image = sitk.JoinSeries(image)
    
    # Write NIfTI
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_path), useCompression=True)
    
    logger.info(f"Saved NIfTI: {output_path.name} ({image.GetSize()})")
    return image


def build_sop_mapping(series: SeriesInfo) -> Dict[str, Any]:
    """Build SOP_UID to slice index mapping for annotation alignment.
    
    Returns a dict with:
    - sop_to_slice: {SOP_UID: slice_index}
    - slice_to_sop: {slice_index: SOP_UID}
    - z_positions: List of Z positions in slice order
    """
    sop_to_slice = {}
    slice_to_sop = {}
    z_positions = []
    
    for slice_info in series.slices:
        sop_to_slice[slice_info.sop_instance_uid] = slice_info.slice_index
        slice_to_sop[slice_info.slice_index] = slice_info.sop_instance_uid
        z_positions.append(slice_info.z_position)
    
    return {
        "sop_to_slice": sop_to_slice,
        "slice_to_sop": slice_to_sop,
        "z_positions": z_positions,
        "num_slices": len(series.slices),
    }


def convert_patient(
    patient_id: str,
    cfg: LIDCConfig,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Convert a single LIDC-IDRI patient to NIfTI.
    
    Args:
        patient_id: Patient ID like "LIDC-IDRI-0001"
        cfg: LIDC configuration
        overwrite: If True, overwrite existing files
        
    Returns:
        Manifest dict with conversion metadata
    """
    patient_dir = cfg.DICOM_ROOT / patient_id
    if not patient_dir.exists():
        raise FileNotFoundError(f"Patient directory not found: {patient_dir}")
    
    case = get_lidc_case(cfg, patient_id)
    
    # Check if already converted
    if case.ct_nifti.exists() and not overwrite:
        logger.info(f"[SKIP] {patient_id}: already converted")
        # Load existing manifest
        if case.manifest_path.exists():
            with open(case.manifest_path) as f:
                return json.load(f)
        return {"status": "skipped", "patient_id": patient_id}
    
    ensure_case_dirs(case)
    
    # Find and group DICOM files
    logger.info(f"[{patient_id}] Scanning DICOM files...")
    dcm_files = find_dicom_files(patient_dir)
    if not dcm_files:
        raise ValueError(f"No DICOM files found in {patient_dir}")
    
    series_map = group_by_series(dcm_files)
    logger.info(f"[{patient_id}] Found {len(series_map)} series, {len(dcm_files)} files")
    
    # Select CT series with DETERMINISTIC logging
    ct_series, selection_log = select_ct_series(series_map, patient_id)
    if ct_series is None:
        raise ValueError(f"No CT series found for {patient_id}: {selection_log.get('selection_reason', 'unknown')}")
    
    logger.info(
        f"[{patient_id}] Selected CT: {ct_series.num_slices} slices, "
        f"thickness={ct_series.slice_thickness}mm, "
        f"desc='{ct_series.series_description}'"
    )
    
    # Convert to NIfTI
    image = convert_series_to_nifti(ct_series, case.ct_nifti)
    
    # Build SOP mapping for annotation alignment
    sop_mapping = build_sop_mapping(ct_series)
    
    # Calculate volume stats (for logging)
    spacing = image.GetSpacing()
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    
    # Build manifest
    manifest = {
        "patient_id": patient_id,
        "status": "converted",
        "ct": {
            "series_uid": ct_series.series_uid,
            "study_uid": ct_series.study_uid,
            "series_description": ct_series.series_description,
            "num_slices": ct_series.num_slices,
            "slice_thickness": ct_series.slice_thickness,
            "pixel_spacing": list(ct_series.pixel_spacing) if ct_series.pixel_spacing else None,
            "size": list(image.GetSize()),
            "spacing": list(image.GetSpacing()),
            "origin": list(image.GetOrigin()),
            "voxel_volume_mm3": voxel_volume_mm3,
        },
        "series_selection_log": selection_log,
        "sop_mapping": sop_mapping,
        "output_files": {
            "ct_nifti": str(case.ct_nifti),
        },
    }
    
    # Save manifest
    with open(case.manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"[{patient_id}] ✓ Conversion complete")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Convert LIDC-IDRI DICOM to NIfTI"
    )
    parser.add_argument(
        "--patient",
        type=str,
        help="Single patient ID (e.g., LIDC-IDRI-0001)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all available patients",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of patients to process (for testing)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing conversions",
    )
    args = parser.parse_args()
    
    cfg = LIDCConfig()
    ensure_lidc_dirs(cfg)
    
    # Determine which patients to process
    if args.patient:
        patient_ids = [args.patient]
    elif args.all:
        patient_ids = list_available_patients(cfg)
        if args.limit:
            patient_ids = patient_ids[:args.limit]
    else:
        parser.error("Specify --patient or --all")
        return
    
    logger.info(f"Processing {len(patient_ids)} patients...")
    
    results = {"success": [], "failed": [], "skipped": []}
    
    for patient_id in tqdm(patient_ids, desc="Converting"):
        try:
            manifest = convert_patient(patient_id, cfg, overwrite=args.overwrite)
            if manifest.get("status") == "skipped":
                results["skipped"].append(patient_id)
            else:
                results["success"].append(patient_id)
        except Exception as e:
            logger.error(f"[{patient_id}] FAILED: {e}")
            results["failed"].append({"patient_id": patient_id, "error": str(e)})
    
    # Summary
    logger.info(
        f"\nSummary: {len(results['success'])} converted, "
        f"{len(results['skipped'])} skipped, "
        f"{len(results['failed'])} failed"
    )
    
    if results["failed"]:
        logger.warning("Failed patients:")
        for f in results["failed"]:
            logger.warning(f"  - {f['patient_id']}: {f['error']}")


if __name__ == "__main__":
    main()
