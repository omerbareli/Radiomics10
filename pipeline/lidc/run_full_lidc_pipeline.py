# pipeline/lidc/run_full_lidc_pipeline.py
"""Unified pipeline runner for full LIDC-IDRI dataset with strict QC.

This script orchestrates:
1. DICOM → NIfTI conversion for all patients
2. XML → voxel-wise nodule mask generation
3. QC flag assignment (OK / QUARANTINE / FAIL)
4. Dataset-wide and sample QC reports

Usage:
    # Test on 10 patients
    python -m pipeline.lidc.run_full_lidc_pipeline --limit 10 --overwrite --seed 42
    
    # Full dataset with resume capability
    python -m pipeline.lidc.run_full_lidc_pipeline --all --resume --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from pipeline.lidc.config import (
    LIDCConfig,
    ensure_case_dirs,
    ensure_lidc_dirs,
    get_lidc_case,
    list_available_patients,
)
from pipeline.lidc.convert_lidc_dicom_to_nifti import convert_patient
from pipeline.lidc.parse_lidc_xml_to_masks import (
    generate_gt_mask_strict,
    load_metadata_nodule_counts,
)
from pipeline.lidc.generate_qc_overlays import generate_qc_overlay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# QC Flag Determination
# ============================================================================

def determine_qc_flag(
    result: Dict[str, Any],
    conversion_manifest: Dict[str, Any],
    z_diff_max_threshold: float = 1.0,
) -> str:
    """Determine QC flag based on gating rules.
    
    Args:
        result: Result from generate_gt_mask_strict
        conversion_manifest: Manifest from convert_patient
        z_diff_max_threshold: Max Z-diff before QUARANTINE (default 1.0mm)
        
    Returns:
        One of: "OK", "QUARANTINE", "FAIL"
    """
    # FAIL conditions
    status = result.get("status", "UNKNOWN")
    if status not in ("SUCCESS",):
        return "FAIL"
    
    roi_stats = result.get("roi_stats", {})
    
    # FAIL: Any ROIs dropped
    if roi_stats.get("rois_dropped", 0) > 0:
        return "FAIL"
    
    # FAIL: No-candidate Z fallbacks (should never happen if mapping worked)
    if roi_stats.get("z_fallback_no_candidate_count", 0) > 0:
        return "FAIL"
    
    # FAIL: Metadata mismatch (study/series UID mismatch)
    metadata_validation = result.get("metadata_validation")
    if metadata_validation == "MISMATCH":
        return "FAIL"
    
    # QUARANTINE conditions
    sop_match_rate = roi_stats.get("sop_match_rate", 1.0)
    z_fallback_rate = roi_stats.get("z_fallback_rate", 0.0)
    z_diff_mm_max = roi_stats.get("z_diff_mm_max", 0.0)
    z_fallback_ambiguous_count = roi_stats.get("z_fallback_ambiguous_count", 0)
    
    if sop_match_rate < 0.90:
        return "QUARANTINE"
    
    if z_fallback_rate > 0.10:
        return "QUARANTINE"
    
    if z_diff_mm_max > z_diff_max_threshold:
        return "QUARANTINE"
    
    if z_fallback_ambiguous_count > 0:
        return "QUARANTINE"
    
    return "OK"


def determine_metadata_validation(
    result: Dict[str, Any],
    conversion_manifest: Dict[str, Any],
) -> str:
    """Determine metadata validation status.
    
    Returns:
        One of: "OK", "MISMATCH", "UNKNOWN"
    """
    # Get UIDs from conversion manifest
    ct_info = conversion_manifest.get("ct", {})
    ct_study_uid = ct_info.get("study_uid", "")
    ct_series_uid = ct_info.get("series_uid", "")
    
    # Get XML UIDs from result (stored during parsing)
    xml_study_uid = result.get("xml_study_uid", "")
    xml_series_uid = result.get("xml_series_uid", "")
    
    # If we don't have XML UIDs, status is UNKNOWN
    if not xml_study_uid or not xml_series_uid:
        return "UNKNOWN"
    
    # Check matches
    study_match = (ct_study_uid == xml_study_uid)
    series_match = (ct_series_uid == xml_series_uid)
    
    if study_match and series_match:
        return "OK"
    else:
        return "MISMATCH"


# ============================================================================
# Per-Patient Processing
# ============================================================================

def process_patient(
    patient_id: str,
    cfg: LIDCConfig,
    expected_counts: Dict[str, int],
    overwrite: bool = False,
    resume: bool = False,
) -> Dict[str, Any]:
    """Process a single patient through the full pipeline.
    
    Returns a row dict for the QC CSV.
    """
    case = get_lidc_case(cfg, patient_id)
    
    # Resume mode: check if already fully processed
    if resume and not overwrite:
        if (case.ct_nifti.exists() and 
            case.nodule_mask_gt.exists() and 
            case.manifest_path.exists()):
            # Load existing results
            try:
                with open(case.manifest_path) as f:
                    manifest = json.load(f)
                # Skip - already processed
                return {"patient_id": patient_id, "status": "SKIPPED_RESUME"}
            except Exception:
                pass  # Re-process if manifest is corrupt
    
    row = {
        "patient_id": patient_id,
        "status": "PENDING",
        "qc_flag": "UNKNOWN",
    }
    
    # Phase A1: DICOM → NIfTI conversion
    conversion_manifest = None
    try:
        conversion_manifest = convert_patient(patient_id, cfg, overwrite=overwrite)
        row["ct_conversion_status"] = "SUCCESS"
    except FileNotFoundError as e:
        row["status"] = "MISSING_DICOM"
        row["qc_flag"] = "FAIL"
        row["error"] = str(e)
        return row
    except Exception as e:
        row["status"] = "CONVERSION_FAILED"
        row["qc_flag"] = "FAIL"
        row["error"] = str(e)
        return row
    
    # Phase A2: XML → Mask generation
    try:
        result = generate_gt_mask_strict(
            patient_id, cfg,
            overwrite=overwrite,
            strict=True,
            expected_nodule_counts=expected_counts,
        )
    except Exception as e:
        result = {
            "status": "EXCEPTION",
            "patient_id": patient_id,
            "error": str(e),
        }
    
    # Build row from results
    status = result.get("status", "UNKNOWN")
    row["status"] = status
    
    # ROI stats
    roi_stats = result.get("roi_stats", {})
    row["roi_total"] = roi_stats.get("total_rois_in_xml", 0)
    row["roi_mapped_by_sop"] = roi_stats.get("rois_mapped_by_sop", 0)
    row["roi_mapped_by_z_fallback"] = roi_stats.get("rois_mapped_by_z", 0)
    row["roi_dropped"] = roi_stats.get("rois_dropped", 0)
    # Set sop_match_rate to NaN when roi_total == 0 to avoid artificial mean reduction
    row["sop_match_rate"] = roi_stats.get("sop_match_rate") if row["roi_total"] > 0 else float('nan')
    row["z_fallback_rate"] = roi_stats.get("z_fallback_rate") if row["roi_total"] > 0 else float('nan')
    
    # Z-diff stats
    row["z_diff_mm_max"] = roi_stats.get("z_diff_mm_max", 0.0)
    row["z_diff_mm_median"] = roi_stats.get("z_diff_mm_median", 0.0)
    row["z_fallback_ambiguous_count"] = roi_stats.get("z_fallback_ambiguous_count", 0)
    row["z_fallback_no_candidate_count"] = roi_stats.get("z_fallback_no_candidate_count", 0)
    row["z_tolerance_mm"] = roi_stats.get("z_tolerance_mm", 0.5)
    
    # Reader/nodule metrics
    row["num_readers"] = roi_stats.get("num_readers", 0)
    row["num_nodules_with_chars"] = roi_stats.get("total_nodules_with_chars", 0)
    row["num_nodules_without_chars"] = roi_stats.get("total_nodules_without_chars", 0)
    
    # Volume metrics
    volume_info = result.get("volume", {})
    row["mask_voxel_count"] = volume_info.get("voxel_count", 0)
    row["voxel_volume_mm3"] = volume_info.get("voxel_volume_mm3", 0.0)
    row["mask_volume_mm3"] = volume_info.get("volume_mm3", 0.0)
    row["mask_volume_ml"] = row["mask_volume_mm3"] / 1000.0
    
    spacing = volume_info.get("spacing_mm", [0, 0, 0])
    row["ct_spacing_x"] = spacing[0] if len(spacing) > 0 else 0.0
    row["ct_spacing_y"] = spacing[1] if len(spacing) > 1 else 0.0
    row["ct_spacing_z"] = spacing[2] if len(spacing) > 2 else 0.0
    
    # Metadata validation
    row["metadata_validation"] = determine_metadata_validation(result, conversion_manifest or {})
    
    # Extract study/series match bools
    ct_info = (conversion_manifest or {}).get("ct", {})
    xml_study_uid = result.get("xml_study_uid", "")
    xml_series_uid = result.get("xml_series_uid", "")
    row["study_uid_match"] = (ct_info.get("study_uid", "") == xml_study_uid) if xml_study_uid else None
    row["series_uid_match"] = (ct_info.get("series_uid", "") == xml_series_uid) if xml_series_uid else None
    
    # Series selection evidence
    selection_log = (conversion_manifest or {}).get("series_selection_log", {})
    row["selected_series_instance_uid"] = ct_info.get("series_uid", "")
    row["selected_study_instance_uid"] = ct_info.get("study_uid", "")
    row["num_ct_series_candidates"] = selection_log.get("ct_candidates", 0)
    row["selected_num_slices"] = ct_info.get("num_slices", 0)
    row["selected_slice_thickness"] = ct_info.get("slice_thickness", 0.0)
    
    # Output paths
    row["ct_path"] = str(case.ct_nifti) if case.ct_nifti.exists() else ""
    row["mask_path"] = str(case.nodule_mask_gt) if case.nodule_mask_gt.exists() else ""
    row["manifest_path"] = str(case.manifest_path) if case.manifest_path.exists() else ""
    
    # Error if present
    if "error" in result:
        row["error"] = result["error"]
    
    # Determine QC flag
    row["qc_flag"] = determine_qc_flag(result, conversion_manifest or {})
    
    return row


# ============================================================================
# Sample Selection
# ============================================================================

def select_sample(
    df: pd.DataFrame,
    seed: int = 42,
    n_ok: int = 30,
    n_quarantine: int = 15,
    n_extremes: int = 10,
) -> List[str]:
    """Select a deterministic sample for QC overlay generation.
    
    Args:
        df: Full QC DataFrame
        seed: Random seed for reproducibility
        n_ok: Number of random OK cases
        n_quarantine: Number of random QUARANTINE cases
        n_extremes: Number of extreme cases (largest/smallest masks, many nodules)
        
    Returns:
        List of patient_ids for the sample
    """
    rng = random.Random(seed)
    sample_ids = set()
    
    # Filter by QC flag
    ok_cases = df[df["qc_flag"] == "OK"]["patient_id"].tolist()
    quarantine_cases = df[df["qc_flag"] == "QUARANTINE"]["patient_id"].tolist()
    fail_cases = df[df["qc_flag"] == "FAIL"]["patient_id"].tolist()
    
    # Random OK samples
    if ok_cases:
        sample_ids.update(rng.sample(ok_cases, min(n_ok, len(ok_cases))))
    
    # Random QUARANTINE samples
    if quarantine_cases:
        sample_ids.update(rng.sample(quarantine_cases, min(n_quarantine, len(quarantine_cases))))
    
    # Include some FAIL cases for debugging (up to 5)
    if fail_cases:
        sample_ids.update(rng.sample(fail_cases, min(5, len(fail_cases))))
    
    # Extremes: largest masks
    if "mask_volume_mm3" in df.columns:
        largest = df.nlargest(n_extremes // 2, "mask_volume_mm3")["patient_id"].tolist()
        sample_ids.update(largest)
        
        # Smallest non-zero masks
        nonzero = df[df["mask_volume_mm3"] > 0]
        if len(nonzero) > 0:
            smallest = nonzero.nsmallest(n_extremes // 2, "mask_volume_mm3")["patient_id"].tolist()
            sample_ids.update(smallest)
    
    # Extremes: most nodules
    if "num_nodules_with_chars" in df.columns:
        most_nodules = df.nlargest(n_extremes // 2, "num_nodules_with_chars")["patient_id"].tolist()
        sample_ids.update(most_nodules)
    
    return sorted(sample_ids)


# ============================================================================
# Sample Summary Generation
# ============================================================================

def generate_sample_summary(df: pd.DataFrame, seed: int) -> Dict[str, Any]:
    """Generate aggregated summary metrics for the sample.
    
    Args:
        df: Sample QC DataFrame
        seed: Random seed used for sampling
        
    Returns:
        Summary dict
    """
    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "sample_size": len(df),
    }
    
    # QC flag counts
    qc_counts = df["qc_flag"].value_counts().to_dict()
    summary["qc_flag_counts"] = {
        "OK": qc_counts.get("OK", 0),
        "QUARANTINE": qc_counts.get("QUARANTINE", 0),
        "FAIL": qc_counts.get("FAIL", 0),
    }
    
    # SOP match rate stats
    if "sop_match_rate" in df.columns:
        sop_rates = df["sop_match_rate"].dropna()
        summary["sop_match_rate"] = {
            "mean": float(sop_rates.mean()) if len(sop_rates) > 0 else None,
            "median": float(sop_rates.median()) if len(sop_rates) > 0 else None,
            "min": float(sop_rates.min()) if len(sop_rates) > 0 else None,
            "max": float(sop_rates.max()) if len(sop_rates) > 0 else None,
        }
    
    # Z fallback rate stats
    if "z_fallback_rate" in df.columns:
        z_rates = df["z_fallback_rate"].dropna()
        summary["z_fallback_rate"] = {
            "mean": float(z_rates.mean()) if len(z_rates) > 0 else None,
            "median": float(z_rates.median()) if len(z_rates) > 0 else None,
            "min": float(z_rates.min()) if len(z_rates) > 0 else None,
            "max": float(z_rates.max()) if len(z_rates) > 0 else None,
        }
    
    # Z-diff max stats
    if "z_diff_mm_max" in df.columns:
        z_diffs = df["z_diff_mm_max"].dropna()
        summary["z_diff_mm_max"] = {
            "max_across_sample": float(z_diffs.max()) if len(z_diffs) > 0 else None,
            "median_across_sample": float(z_diffs.median()) if len(z_diffs) > 0 else None,
        }
    
    # Z fallback ambiguous stats
    if "z_fallback_ambiguous_count" in df.columns:
        ambiguous = df["z_fallback_ambiguous_count"]
        summary["z_fallback_ambiguous"] = {
            "total_count": int(ambiguous.sum()),
            "pct_cases_with_ambiguity": float((ambiguous > 0).mean() * 100) if len(ambiguous) > 0 else 0.0,
        }
    
    # Metadata validation stats
    if "metadata_validation" in df.columns:
        mv_counts = df["metadata_validation"].value_counts()
        total = len(df)
        summary["metadata_validation"] = {
            "OK_pct": float(mv_counts.get("OK", 0) / total * 100) if total > 0 else 0.0,
            "MISMATCH_pct": float(mv_counts.get("MISMATCH", 0) / total * 100) if total > 0 else 0.0,
            "UNKNOWN_pct": float(mv_counts.get("UNKNOWN", 0) / total * 100) if total > 0 else 0.0,
        }
    
    # Mask volume stats
    for col in ["mask_voxel_count", "mask_volume_mm3", "mask_volume_ml"]:
        if col in df.columns:
            vals = df[col].dropna()
            summary[col] = {
                "min": float(vals.min()) if len(vals) > 0 else None,
                "median": float(vals.median()) if len(vals) > 0 else None,
                "max": float(vals.max()) if len(vals) > 0 else None,
            }
    
    # Reader distribution
    if "num_readers" in df.columns:
        readers = df["num_readers"].dropna().astype(int)
        dist = readers.value_counts().to_dict()
        summary["num_readers"] = {
            "distribution": {str(k): v for k, v in sorted(dist.items())},
            "pct_gte_3_readers": float((readers >= 3).mean() * 100) if len(readers) > 0 else 0.0,
        }
    
    return summary


# ============================================================================
# Atomic Write Helpers
# ============================================================================

def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write CSV atomically (temp file → rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".csv.tmp")
    df.to_csv(tmp_path, index=False)
    shutil.move(str(tmp_path), str(path))


def atomic_write_json(data: Any, path: Path) -> None:
    """Write JSON atomically (temp file → rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    shutil.move(str(tmp_path), str(path))


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full LIDC-IDRI pipeline with strict QC"
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
        help="Limit number of patients to process",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force reprocessing of all patients",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip patients that already have valid outputs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sample selection (default: 42)",
    )
    parser.add_argument(
        "--skip-overlays",
        action="store_true",
        help="Skip overlay generation for sample cases",
    )
    parser.add_argument(
        "--patient-ids-file",
        type=str,
        default=None,
        help="Path to file with patient IDs (one per line) to process",
    )
    args = parser.parse_args()
    
    # Validate arguments
    if args.patient_ids_file:
        # When using patient-ids-file, don't require --all or --limit
        pass
    elif not args.all and args.limit is None:
        parser.error("Specify --all or --limit N or --patient-ids-file")
        return
    
    cfg = LIDCConfig()
    ensure_lidc_dirs(cfg)
    
    # Determine patients to process
    if args.patient_ids_file:
        # Read patient IDs from file
        with open(args.patient_ids_file) as f:
            patient_ids = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(patient_ids)} patient IDs from {args.patient_ids_file}")
    else:
        patient_ids = list_available_patients(cfg)
        if args.limit:
            patient_ids = patient_ids[:args.limit]
    
    logger.info(f"Pipeline starting: {len(patient_ids)} patients")
    logger.info(f"  --overwrite={args.overwrite}, --resume={args.resume}, --seed={args.seed}")
    
    # Load metadata for cross-reference
    expected_counts = load_metadata_nodule_counts(cfg)
    
    # Track missing cases (DICOM not found)
    missing_cases = []
    
    # Process all patients
    rows = []
    for patient_id in tqdm(patient_ids, desc="Processing"):
        try:
            row = process_patient(
                patient_id, cfg, expected_counts,
                overwrite=args.overwrite,
                resume=args.resume,
            )
            
            if row.get("status") == "MISSING_DICOM":
                missing_cases.append(patient_id)
            
            rows.append(row)
            
        except Exception as e:
            logger.error(f"[{patient_id}] EXCEPTION: {e}")
            rows.append({
                "patient_id": patient_id,
                "status": "EXCEPTION",
                "qc_flag": "FAIL",
                "error": str(e),
            })
    
    # Build full QC DataFrame
    df_new = pd.DataFrame(rows)
    
    # Phase C1: Write dataset-wide CSV
    full_csv_path = cfg.QC_DIR / "lidc_full_qc_report.csv"
    
    # When using --patient-ids-file, merge with existing QC report
    if args.patient_ids_file and full_csv_path.exists():
        df_existing = pd.read_csv(full_csv_path)
        # Remove old rows for patients we just processed
        df_existing = df_existing[~df_existing["patient_id"].isin(df_new["patient_id"])]
        # Merge with new results
        df = pd.concat([df_existing, df_new], ignore_index=True)
        # Sort by patient_id for consistency
        df = df.sort_values("patient_id").reset_index(drop=True)
        logger.info(f"Merged {len(df_new)} updated rows into existing QC report")
    else:
        df = df_new
    
    atomic_write_csv(df, full_csv_path)
    logger.info(f"✓ Full QC report: {full_csv_path} ({len(df)} rows)")
    
    # Phase C2: Save missing cases
    missing_path = cfg.QC_DIR / "missing_cases.txt"
    missing_path.write_text("\n".join(missing_cases) + "\n" if missing_cases else "")
    logger.info(f"✓ Missing cases: {len(missing_cases)} → {missing_path}")
    
    # Filter to non-skipped for sample selection
    df_processed = df[~df["status"].isin(["SKIPPED_RESUME", "MISSING_DICOM"])]
    
    # Phase C2: Select sample
    sample_ids = select_sample(df_processed, seed=args.seed)
    
    # Save sample IDs for reproducibility
    sample_ids_path = cfg.QC_DIR / "lidc_sample_ids.json"
    atomic_write_json({"seed": args.seed, "sample_ids": sample_ids}, sample_ids_path)
    logger.info(f"✓ Sample IDs: {len(sample_ids)} → {sample_ids_path}")
    
    # Phase C3: Generate overlays for sample cases
    if not args.skip_overlays:
        logger.info(f"Generating overlays for {len(sample_ids)} sample cases...")
        overlay_results = []
        for pid in tqdm(sample_ids, desc="Overlays"):
            try:
                result = generate_qc_overlay(pid, cfg, overwrite=args.overwrite)
                overlay_results.append(result)
            except Exception as e:
                logger.warning(f"[{pid}] Overlay failed: {e}")
        
        overlay_success = sum(1 for r in overlay_results if r.get("status") == "SUCCESS")
        logger.info(f"✓ Overlays: {overlay_success}/{len(sample_ids)} generated")
    
    # Phase C4: Generate sample CSV
    df_sample = df[df["patient_id"].isin(sample_ids)].copy()
    
    # Add overlay path column
    df_sample["overlay_path"] = df_sample["patient_id"].apply(
        lambda pid: str(get_lidc_case(cfg, pid).qc_dir / "qc_overlay.png")
    )
    
    sample_csv_path = cfg.QC_DIR / "lidc_sample_qc_report.csv"
    atomic_write_csv(df_sample, sample_csv_path)
    logger.info(f"✓ Sample QC report: {sample_csv_path} ({len(df_sample)} rows)")
    
    # Phase C5: Generate sample summary
    summary = generate_sample_summary(df_sample, seed=args.seed)
    summary_path = cfg.QC_DIR / "lidc_sample_summary.json"
    atomic_write_json(summary, summary_path)
    logger.info(f"✓ Sample summary: {summary_path}")
    
    # Final summary
    qc_counts = df["qc_flag"].value_counts()
    logger.info(
        f"\n{'='*50}\n"
        f"PIPELINE COMPLETE\n"
        f"{'='*50}\n"
        f"Total processed: {len(df)}\n"
        f"  OK:         {qc_counts.get('OK', 0)}\n"
        f"  QUARANTINE: {qc_counts.get('QUARANTINE', 0)}\n"
        f"  FAIL:       {qc_counts.get('FAIL', 0)}\n"
        f"  UNKNOWN:    {qc_counts.get('UNKNOWN', 0)}\n"
        f"Missing DICOM: {len(missing_cases)}\n"
        f"Sample size: {len(sample_ids)}\n"
    )
    
    # Print warning thresholds check
    if len(df_processed) > 0:
        mean_sop = df_processed["sop_match_rate"].mean() if "sop_match_rate" in df_processed.columns else 0
        mean_z_fall = df_processed["z_fallback_rate"].mean() if "z_fallback_rate" in df_processed.columns else 0
        logger.info(
            f"Dataset health (warning thresholds):\n"
            f"  SOP match rate mean: {mean_sop:.1%} (target >95%)\n"
            f"  Z-fallback rate mean: {mean_z_fall:.1%} (target <5%)\n"
        )


if __name__ == "__main__":
    main()
