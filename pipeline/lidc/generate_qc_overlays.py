# pipeline/lidc/generate_qc_overlays.py
"""Generate QC overlay montages for LIDC-IDRI CT and mask visualization.

This script creates PNG montages showing CT slices with nodule masks overlaid
in semi-transparent red to visually verify orientation and alignment.

Output:
- Per-patient: output/cases/LIDC-IDRI-XXXX/qc/qc_overlay.png
- Global: output/qc/qc_overlay_LIDC-IDRI-XXXX.png

Usage:
    python -m pipeline.lidc.generate_qc_overlays --patient LIDC-IDRI-0001
    python -m pipeline.lidc.generate_qc_overlays --all --limit 20
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for PNG generation
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from pipeline.lidc.config import (
    LIDCConfig,
    LIDCCasePaths,
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


def window_ct(
    ct_array: np.ndarray,
    window_center: float = -400,
    window_width: float = 1500,
) -> np.ndarray:
    """Apply lung window to CT image.
    
    Args:
        ct_array: CT HU values
        window_center: Center of window
        window_width: Width of window
        
    Returns:
        Normalized array [0, 1]
    """
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2
    windowed = np.clip(ct_array, lower, upper)
    normalized = (windowed - lower) / (upper - lower)
    return normalized


def find_mask_slices(mask_array: np.ndarray, margin: int = 1) -> List[int]:
    """Find slice indices containing mask voxels.
    
    Args:
        mask_array: Binary mask (Z, Y, X)
        margin: Extra slices before/after mask region
        
    Returns:
        List of slice indices
    """
    mask_slices = []
    for z in range(mask_array.shape[0]):
        if np.any(mask_array[z] > 0):
            mask_slices.append(z)
    
    if not mask_slices:
        return []
    
    # Add margin
    min_z = max(0, min(mask_slices) - margin)
    max_z = min(mask_array.shape[0] - 1, max(mask_slices) + margin)
    
    return list(range(min_z, max_z + 1))


def select_representative_slices(
    mask_slices: List[int],
    max_slices: int = 16,
) -> List[int]:
    """Select representative slices for montage.
    
    Args:
        mask_slices: All slices with mask
        max_slices: Maximum number of slices to show
        
    Returns:
        Selected slice indices
    """
    if len(mask_slices) <= max_slices:
        return mask_slices
    
    # Uniform sampling
    indices = np.linspace(0, len(mask_slices) - 1, max_slices, dtype=int)
    return [mask_slices[i] for i in indices]


def create_overlay_slice(
    ct_slice: np.ndarray,
    mask_slice: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Create RGB overlay of CT and mask.
    
    Args:
        ct_slice: Windowed CT slice (normalized 0-1)
        mask_slice: Binary mask slice
        alpha: Mask transparency
        
    Returns:
        RGB image (H, W, 3)
    """
    # Create grayscale CT background
    ct_rgb = np.stack([ct_slice] * 3, axis=-1)
    
    # Create red mask overlay
    mask_rgb = np.zeros((*mask_slice.shape, 3))
    mask_rgb[mask_slice > 0] = [1, 0, 0]  # Red
    
    # Blend
    mask_bool = mask_slice > 0
    ct_rgb[mask_bool] = (1 - alpha) * ct_rgb[mask_bool] + alpha * mask_rgb[mask_bool]
    
    return ct_rgb


def create_montage(
    ct_array: np.ndarray,
    mask_array: np.ndarray,
    slice_indices: List[int],
    ncols: int = 4,
    figsize_per_slice: float = 3.0,
) -> plt.Figure:
    """Create montage figure showing CT with mask overlay.
    
    Args:
        ct_array: CT volume (Z, Y, X)
        mask_array: Mask volume (Z, Y, X)
        slice_indices: Slices to show
        ncols: Number of columns
        figsize_per_slice: Size per subplot in inches
        
    Returns:
        Matplotlib figure
    """
    nslices = len(slice_indices)
    nrows = (nslices + ncols - 1) // ncols
    
    figsize = (ncols * figsize_per_slice, nrows * figsize_per_slice)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Flatten axes for easy iteration
    if nslices == 1:
        axes = [axes]
    else:
        axes = axes.flat
    
    # Window CT
    ct_windowed = window_ct(ct_array)
    
    for idx, (ax, z) in enumerate(zip(axes, slice_indices)):
        overlay = create_overlay_slice(ct_windowed[z], mask_array[z])
        ax.imshow(overlay, origin="upper")
        ax.set_title(f"Slice {z}", fontsize=10)
        ax.axis("off")
    
    # Hide unused subplots
    for i in range(nslices, len(list(axes)[:nrows * ncols])):
        if i < len(list(axes)):
            list(axes)[i].axis("off")
            list(axes)[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def generate_qc_overlay(
    patient_id: str,
    cfg: LIDCConfig,
    overwrite: bool = False,
    max_slices: int = 16,
) -> dict:
    """Generate QC overlay for a patient.
    
    Args:
        patient_id: Patient ID
        cfg: Configuration
        overwrite: Overwrite existing overlays
        max_slices: Maximum slices in montage
        
    Returns:
        Result dict
    """
    case = get_lidc_case(cfg, patient_id)
    
    # Output paths
    case_qc_path = case.qc_dir / "qc_overlay.png"
    global_qc_path = cfg.QC_DIR / f"qc_overlay_{patient_id}.png"
    
    # Check prerequisites
    if not case.ct_nifti.exists():
        return {"status": "FAILED", "patient_id": patient_id, "error": "CT not found"}
    
    if not case.nodule_mask_gt.exists():
        return {"status": "FAILED", "patient_id": patient_id, "error": "Mask not found"}
    
    # Check if already done
    if case_qc_path.exists() and global_qc_path.exists() and not overwrite:
        logger.info(f"[SKIP] {patient_id}: overlay already exists")
        return {"status": "SKIPPED", "patient_id": patient_id}
    
    ensure_case_dirs(case)
    
    # Load data
    ct_image = sitk.ReadImage(str(case.ct_nifti))
    ct_array = sitk.GetArrayFromImage(ct_image)  # (Z, Y, X)
    
    mask_image = sitk.ReadImage(str(case.nodule_mask_gt))
    mask_array = sitk.GetArrayFromImage(mask_image)  # (Z, Y, X)
    
    # Find slices with mask
    mask_slices = find_mask_slices(mask_array)
    
    if not mask_slices:
        logger.warning(f"[{patient_id}] No mask voxels found - creating empty QC")
        # Still create a QC showing some middle slices
        mid = ct_array.shape[0] // 2
        mask_slices = list(range(max(0, mid - 4), min(ct_array.shape[0], mid + 4)))
    
    # Select representative slices
    selected_slices = select_representative_slices(mask_slices, max_slices)
    
    # Create montage
    fig = create_montage(ct_array, mask_array, selected_slices)
    
    # Add title with patient info
    mask_voxels = int(np.sum(mask_array > 0))
    spacing = ct_image.GetSpacing()
    volume_mm3 = mask_voxels * spacing[0] * spacing[1] * spacing[2]
    
    fig.suptitle(
        f"{patient_id} | Slices: {len(mask_slices)} | "
        f"Volume: {volume_mm3:.1f}mm³ ({mask_voxels} voxels)",
        fontsize=12, y=1.02,
    )
    
    # Save to both locations
    fig.savefig(case_qc_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(global_qc_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    
    logger.info(
        f"[{patient_id}] ✓ QC overlay saved: "
        f"{len(selected_slices)} slices, {mask_voxels} voxels"
    )
    
    return {
        "status": "SUCCESS",
        "patient_id": patient_id,
        "slices_shown": len(selected_slices),
        "total_mask_slices": len(mask_slices),
        "mask_voxels": mask_voxels,
        "volume_mm3": volume_mm3,
        "output_paths": [str(case_qc_path), str(global_qc_path)],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate QC overlay montages for LIDC-IDRI"
    )
    parser.add_argument(
        "--patient",
        type=str,
        help="Single patient ID (e.g., LIDC-IDRI-0001)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all patients with masks",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of patients",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing overlays",
    )
    parser.add_argument(
        "--max-slices",
        type=int,
        default=16,
        help="Maximum slices per montage (default: 16)",
    )
    args = parser.parse_args()
    
    cfg = LIDCConfig()
    ensure_lidc_dirs(cfg)
    
    # Determine patients
    if args.patient:
        patient_ids = [args.patient]
    elif args.all:
        # Only process patients with masks
        available = list_available_patients(cfg)
        patient_ids = []
        for pid in available:
            case = get_lidc_case(cfg, pid)
            if case.nodule_mask_gt.exists():
                patient_ids.append(pid)
        
        if args.limit:
            patient_ids = patient_ids[:args.limit]
    else:
        parser.error("Specify --patient or --all")
        return
    
    logger.info(f"Generating QC overlays for {len(patient_ids)} patients...")
    
    results = {"SUCCESS": [], "FAILED": [], "SKIPPED": []}
    
    for patient_id in tqdm(patient_ids, desc="Generating QC"):
        try:
            result = generate_qc_overlay(
                patient_id, cfg,
                overwrite=args.overwrite,
                max_slices=args.max_slices,
            )
            status = result.get("status", "UNKNOWN")
            
            if status in results:
                results[status].append(result)
            else:
                results["FAILED"].append(result)
                
        except Exception as e:
            logger.error(f"[{patient_id}] EXCEPTION: {e}")
            results["FAILED"].append({
                "patient_id": patient_id,
                "status": "FAILED",
                "error": str(e),
            })
    
    # Summary
    logger.info(
        f"\n=== SUMMARY ===\n"
        f"SUCCESS: {len(results['SUCCESS'])}\n"
        f"FAILED: {len(results['FAILED'])}\n"
        f"SKIPPED: {len(results['SKIPPED'])}"
    )
    
    if results["SUCCESS"]:
        logger.info(f"\nQC overlays saved to: {cfg.QC_DIR}")
    
    # Save results
    results_path = cfg.QC_DIR / "qc_overlay_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
