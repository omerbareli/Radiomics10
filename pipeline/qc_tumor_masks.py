# pipeline/qc_tumor_masks.py
"""
Comprehensive QC for MedSAM2 tumor masks.

Run:
    python -m pipeline.qc_tumor_masks --output-dir output

Generates:
    - qc_tumor_masks_detailed.csv: Per-patient metrics
    - qc_flags.csv: Flagged cases with reasons
    - qc_tumor_masks_summary.md: Cohort statistics
    - cases/{pid}/qc/tumor_qc_montage.png: Visual QC for flagged cases
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from tqdm import tqdm

from pipeline.config import Config


# ============================================================================
# Configuration
# ============================================================================

HU_HARD_MIN = -1024
HU_HARD_MAX = 3071
OUTSIDE_LUNG_THRESHOLD = 0.10  # Flag if >10% of tumor outside lung
SPACING_TOLERANCE = 0.01  # mm
VOLUME_MIN_ML = 0.1
VOLUME_MAX_ML = 500.0
Z_SCORE_THRESHOLD = 3.0
SPIKINESS_THRESHOLD = 5.0  # max/median slice area ratio


@dataclass
class QCResult:
    patient_id: str
    status: str = "ok"
    flags: List[str] = field(default_factory=list)
    
    # File paths
    ct_path: str = ""
    mask_path: str = ""
    lung_mask_path: str = ""
    suv_path: str = ""
    
    # Geometry
    mask_shape: tuple = ()
    ct_shape: tuple = ()
    shape_match: bool = True
    spacing_match: bool = True
    direction_match: bool = True
    origin_match: bool = True
    spacing: tuple = ()
    
    # Basic mask stats
    tumor_voxels: int = 0
    tumor_vol_ml: float = 0.0
    n_components: int = 0
    
    # Bounding box
    bbox_z: int = 0
    bbox_y: int = 0
    bbox_x: int = 0
    
    # Centroid
    centroid_z: float = 0.0
    centroid_y: float = 0.0
    centroid_x: float = 0.0
    
    # HU stats
    hu_mean: float = np.nan
    hu_std: float = np.nan
    hu_min: float = np.nan
    hu_p01: float = np.nan
    hu_p50: float = np.nan
    hu_p99: float = np.nan
    hu_max: float = np.nan
    hu_has_nan: bool = False
    hu_out_of_range: bool = False
    
    # SUV stats (optional)
    suv_mean: float = np.nan
    suv_std: float = np.nan
    suv_p50: float = np.nan
    suv_max: float = np.nan
    
    # Lung overlap
    lung_mask_available: bool = False
    lung_overlap_fraction: float = np.nan
    outside_lung_voxels: int = 0
    outside_lung_fraction: float = 0.0
    
    # Slice-wise metrics
    z_span: int = 0
    slice_area_max: float = 0.0
    slice_area_median: float = 0.0
    slice_area_std: float = 0.0
    spikiness: float = 0.0
    fill_ratio_mean: float = 0.0
    fill_ratio_min: float = 0.0
    
    # Cross-reference
    boxes_used: int = 0
    missing_sop: int = 0
    
    # Error info
    error: str = ""
    
    def add_flag(self, flag: str):
        if flag not in self.flags:
            self.flags.append(flag)


def load_nifti(path: Path) -> Optional[sitk.Image]:
    """Load NIfTI, return None on failure."""
    try:
        return sitk.ReadImage(str(path))
    except Exception:
        return None


def get_affine_components(img: sitk.Image) -> dict:
    """Extract affine components for comparison."""
    return {
        "spacing": img.GetSpacing(),
        "origin": img.GetOrigin(),
        "direction": img.GetDirection(),
    }


def check_affine_match(img1: sitk.Image, img2: sitk.Image, tol: float = SPACING_TOLERANCE) -> dict:
    """Check if two images have matching affine."""
    a1 = get_affine_components(img1)
    a2 = get_affine_components(img2)
    
    spacing_match = all(
        abs(s1 - s2) < tol for s1, s2 in zip(a1["spacing"], a2["spacing"])
    )
    origin_match = all(
        abs(o1 - o2) < tol for o1, o2 in zip(a1["origin"], a2["origin"])
    )
    direction_match = all(
        abs(d1 - d2) < 1e-6 for d1, d2 in zip(a1["direction"], a2["direction"])
    )
    
    return {
        "spacing_match": spacing_match,
        "origin_match": origin_match,
        "direction_match": direction_match,
    }


def compute_slice_metrics(mask_arr: np.ndarray) -> dict:
    """Compute slice-wise QC metrics."""
    z_slices_with_mask = []
    slice_areas = []
    fill_ratios = []
    
    for z in range(mask_arr.shape[0]):
        slice_2d = mask_arr[z, :, :]
        area = np.sum(slice_2d > 0)
        if area > 0:
            z_slices_with_mask.append(z)
            slice_areas.append(area)
            
            # Compute fill ratio for this slice
            ys, xs = np.where(slice_2d > 0)
            if len(ys) > 0:
                bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
                fill_ratios.append(area / bbox_area if bbox_area > 0 else 0)
    
    if not slice_areas:
        return {
            "z_span": 0,
            "slice_area_max": 0.0,
            "slice_area_median": 0.0,
            "slice_area_std": 0.0,
            "spikiness": 0.0,
            "fill_ratio_mean": 0.0,
            "fill_ratio_min": 0.0,
        }
    
    slice_areas = np.array(slice_areas)
    median_area = np.median(slice_areas)
    spikiness = slice_areas.max() / median_area if median_area > 0 else 0.0
    
    return {
        "z_span": len(z_slices_with_mask),
        "slice_area_max": float(slice_areas.max()),
        "slice_area_median": float(median_area),
        "slice_area_std": float(np.std(slice_areas)),
        "spikiness": float(spikiness),
        "fill_ratio_mean": float(np.mean(fill_ratios)) if fill_ratios else 0.0,
        "fill_ratio_min": float(np.min(fill_ratios)) if fill_ratios else 0.0,
    }


def compute_lung_overlap(mask_arr: np.ndarray, lung_arr: np.ndarray) -> dict:
    """Compute tumor-lung overlap metrics."""
    tumor_volume = np.sum(mask_arr > 0)
    if tumor_volume == 0:
        return {
            "lung_overlap_fraction": np.nan,
            "outside_lung_voxels": 0,
            "outside_lung_fraction": 0.0,
        }
    
    overlap = np.sum((mask_arr > 0) & (lung_arr > 0))
    outside = tumor_volume - overlap
    
    return {
        "lung_overlap_fraction": overlap / tumor_volume,
        "outside_lung_voxels": int(outside),
        "outside_lung_fraction": outside / tumor_volume,
    }


def compute_hu_stats(ct_arr: np.ndarray, mask_arr: np.ndarray) -> dict:
    """Compute HU statistics within mask."""
    hu_vals = ct_arr[mask_arr > 0]
    
    if len(hu_vals) == 0:
        return {
            "hu_mean": np.nan,
            "hu_std": np.nan,
            "hu_min": np.nan,
            "hu_p01": np.nan,
            "hu_p50": np.nan,
            "hu_p99": np.nan,
            "hu_max": np.nan,
            "hu_has_nan": False,
            "hu_out_of_range": False,
        }
    
    has_nan = np.any(np.isnan(hu_vals)) or np.any(np.isinf(hu_vals))
    hu_vals_clean = hu_vals[np.isfinite(hu_vals)]
    
    if len(hu_vals_clean) == 0:
        return {
            "hu_mean": np.nan,
            "hu_std": np.nan,
            "hu_min": np.nan,
            "hu_p01": np.nan,
            "hu_p50": np.nan,
            "hu_p99": np.nan,
            "hu_max": np.nan,
            "hu_has_nan": True,
            "hu_out_of_range": True,
        }
    
    p01 = np.percentile(hu_vals_clean, 1)
    p99 = np.percentile(hu_vals_clean, 99)
    out_of_range = p01 < HU_HARD_MIN or p99 > HU_HARD_MAX
    
    return {
        "hu_mean": float(np.mean(hu_vals_clean)),
        "hu_std": float(np.std(hu_vals_clean)),
        "hu_min": float(np.min(hu_vals_clean)),
        "hu_p01": float(p01),
        "hu_p50": float(np.median(hu_vals_clean)),
        "hu_p99": float(p99),
        "hu_max": float(np.max(hu_vals_clean)),
        "hu_has_nan": has_nan,
        "hu_out_of_range": out_of_range,
    }


def compute_suv_stats(suv_arr: np.ndarray, mask_arr: np.ndarray) -> dict:
    """Compute SUV statistics within mask."""
    suv_vals = suv_arr[mask_arr > 0]
    
    if len(suv_vals) == 0:
        return {"suv_mean": np.nan, "suv_std": np.nan, "suv_p50": np.nan, "suv_max": np.nan}
    
    suv_vals_clean = suv_vals[np.isfinite(suv_vals)]
    if len(suv_vals_clean) == 0:
        return {"suv_mean": np.nan, "suv_std": np.nan, "suv_p50": np.nan, "suv_max": np.nan}
    
    return {
        "suv_mean": float(np.mean(suv_vals_clean)),
        "suv_std": float(np.std(suv_vals_clean)),
        "suv_p50": float(np.median(suv_vals_clean)),
        "suv_max": float(np.max(suv_vals_clean)),
    }


def generate_montage(
    ct_arr: np.ndarray,
    mask_arr: np.ndarray,
    suv_arr: Optional[np.ndarray],
    result: QCResult,
    output_path: Path,
    window_center: float = -400,
    window_width: float = 1500,
):
    """Generate 3-panel QC montage with overlay."""
    # Get centroid slice indices
    cz = int(result.centroid_z)
    cy = int(result.centroid_y)
    cx = int(result.centroid_x)
    
    # Clamp to valid range
    cz = max(0, min(cz, ct_arr.shape[0] - 1))
    cy = max(0, min(cy, ct_arr.shape[1] - 1))
    cx = max(0, min(cx, ct_arr.shape[2] - 1))
    
    # Window CT
    ct_min = window_center - window_width / 2
    ct_max = window_center + window_width / 2
    
    def window_ct(arr):
        return np.clip((arr - ct_min) / (ct_max - ct_min), 0, 1)
    
    # Get slices
    axial_ct = window_ct(ct_arr[cz, :, :])
    axial_mask = mask_arr[cz, :, :] > 0
    
    coronal_ct = window_ct(ct_arr[:, cy, :])
    coronal_mask = mask_arr[:, cy, :] > 0
    
    sagittal_ct = window_ct(ct_arr[:, :, cx])
    sagittal_mask = mask_arr[:, :, cx] > 0
    
    # Create figure
    n_cols = 4 if suv_arr is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5))
    
    # Axial
    axes[0].imshow(axial_ct, cmap='gray', aspect='auto')
    axes[0].imshow(axial_mask, cmap='Reds', alpha=0.4 * axial_mask.astype(float), aspect='auto')
    axes[0].set_title(f'Axial (z={cz})')
    axes[0].axis('off')
    
    # Coronal
    axes[1].imshow(coronal_ct, cmap='gray', aspect='auto')
    axes[1].imshow(coronal_mask, cmap='Reds', alpha=0.4 * coronal_mask.astype(float), aspect='auto')
    axes[1].set_title(f'Coronal (y={cy})')
    axes[1].axis('off')
    
    # Sagittal
    axes[2].imshow(sagittal_ct, cmap='gray', aspect='auto')
    axes[2].imshow(sagittal_mask, cmap='Reds', alpha=0.4 * sagittal_mask.astype(float), aspect='auto')
    axes[2].set_title(f'Sagittal (x={cx})')
    axes[2].axis('off')
    
    # SUV if available
    if suv_arr is not None and n_cols == 4:
        axial_suv = suv_arr[cz, :, :]
        suv_max = np.percentile(axial_suv[axial_suv > 0], 99) if np.any(axial_suv > 0) else 10
        axes[3].imshow(axial_suv, cmap='hot', vmin=0, vmax=suv_max, aspect='auto')
        axes[3].imshow(axial_mask, cmap='Blues', alpha=0.3 * axial_mask.astype(float), aspect='auto')
        axes[3].set_title(f'SUV (z={cz})')
        axes[3].axis('off')
    
    # Add metrics text box
    flags_str = ", ".join(result.flags) if result.flags else "none"
    metrics_text = (
        f"Patient: {result.patient_id}\n"
        f"Volume: {result.tumor_vol_ml:.1f} ml ({result.tumor_voxels} vox)\n"
        f"Components: {result.n_components}\n"
        f"HU mean: {result.hu_mean:.1f}, p50: {result.hu_p50:.1f}\n"
        f"Z-span: {result.z_span} slices\n"
        f"Spikiness: {result.spikiness:.2f}\n"
        f"Flags: {flags_str}"
    )
    fig.text(0.02, 0.02, metrics_text, fontsize=8, family='monospace',
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def qc_single_patient(
    patient_id: str,
    index_row: pd.Series,
    cases_dir: Path,
) -> QCResult:
    """Run QC on a single patient."""
    result = QCResult(patient_id=patient_id)
    
    # Check status from index
    status = index_row.get("status", "")
    if status in ("missing_ct", "missing_annotations_dir"):
        result.status = status
        result.add_flag(status)
        return result
    
    # Get paths
    mask_path_str = index_row.get("tumor_mask_path", "")
    if not mask_path_str or pd.isna(mask_path_str):
        result.status = "missing_mask_path"
        result.add_flag("missing_mask_path")
        return result
    
    mask_path = Path(mask_path_str)
    result.mask_path = str(mask_path)
    
    # Derive CT path from mask path
    case_dir = mask_path.parent.parent
    ct_path = case_dir / "nifti" / "ct.nii.gz"
    result.ct_path = str(ct_path)
    
    lung_mask_path = case_dir / "seg" / "lung_mask.nii.gz"
    result.lung_mask_path = str(lung_mask_path)
    
    suv_path = case_dir / "nifti" / "pet_suv_ctspace.nii.gz"
    result.suv_path = str(suv_path) if suv_path.exists() else ""
    
    # Cross-reference info from index
    result.boxes_used = int(index_row.get("boxes_used", 0) or 0)
    result.missing_sop = int(index_row.get("missing_sop", 0) or 0)
    
    # Load mask
    if not mask_path.exists():
        result.status = "mask_not_found"
        result.add_flag("mask_not_found")
        result.error = f"Mask file not found: {mask_path}"
        return result
    
    mask_img = load_nifti(mask_path)
    if mask_img is None:
        result.status = "mask_load_error"
        result.add_flag("mask_load_error")
        result.error = "Failed to load mask NIfTI"
        return result
    
    mask_arr = sitk.GetArrayFromImage(mask_img)
    result.mask_shape = mask_arr.shape
    result.spacing = mask_img.GetSpacing()
    
    # Load CT
    if not ct_path.exists():
        result.status = "ct_not_found"
        result.add_flag("ct_not_found")
        result.error = f"CT file not found: {ct_path}"
        return result
    
    ct_img = load_nifti(ct_path)
    if ct_img is None:
        result.status = "ct_load_error"
        result.add_flag("ct_load_error")
        result.error = "Failed to load CT NIfTI"
        return result
    
    ct_arr = sitk.GetArrayFromImage(ct_img)
    result.ct_shape = ct_arr.shape
    
    # Shape check
    result.shape_match = mask_arr.shape == ct_arr.shape
    if not result.shape_match:
        result.add_flag("shape_mismatch")
    
    # Affine check
    affine_check = check_affine_match(mask_img, ct_img)
    result.spacing_match = affine_check["spacing_match"]
    result.origin_match = affine_check["origin_match"]
    result.direction_match = affine_check["direction_match"]
    
    if not result.spacing_match:
        result.add_flag("spacing_mismatch")
    if not result.origin_match:
        result.add_flag("origin_mismatch")
    if not result.direction_match:
        result.add_flag("direction_mismatch")
    
    # Basic mask stats
    result.tumor_voxels = int(np.sum(mask_arr > 0))
    voxel_vol_ml = np.prod(result.spacing) / 1000.0  # mm^3 to ml
    result.tumor_vol_ml = result.tumor_voxels * voxel_vol_ml
    
    # Check for empty mask
    if result.tumor_voxels == 0:
        result.status = "empty_mask"
        result.add_flag("empty_mask")
        return result
    
    # Connected components
    labeled, n_components = ndimage.label(mask_arr > 0)
    result.n_components = n_components
    
    # Bounding box
    zs, ys, xs = np.where(mask_arr > 0)
    result.bbox_z = int(zs.max() - zs.min() + 1)
    result.bbox_y = int(ys.max() - ys.min() + 1)
    result.bbox_x = int(xs.max() - xs.min() + 1)
    
    # Centroid
    result.centroid_z = float(np.mean(zs))
    result.centroid_y = float(np.mean(ys))
    result.centroid_x = float(np.mean(xs))
    
    # HU stats
    hu_stats = compute_hu_stats(ct_arr, mask_arr)
    for k, v in hu_stats.items():
        setattr(result, k, v)
    
    if result.hu_has_nan:
        result.add_flag("hu_has_nan")
    if result.hu_out_of_range:
        result.add_flag("hu_out_of_range")
    
    # Volume flags
    if result.tumor_vol_ml < VOLUME_MIN_ML:
        result.add_flag("very_small_volume")
    if result.tumor_vol_ml > VOLUME_MAX_ML:
        result.add_flag("very_large_volume")
    
    # Slice-wise metrics
    slice_metrics = compute_slice_metrics(mask_arr)
    for k, v in slice_metrics.items():
        setattr(result, k, v)
    
    if result.spikiness > SPIKINESS_THRESHOLD:
        result.add_flag("high_spikiness")
    
    # Lung overlap
    if lung_mask_path.exists():
        lung_img = load_nifti(lung_mask_path)
        if lung_img is not None:
            lung_arr = sitk.GetArrayFromImage(lung_img)
            if lung_arr.shape == mask_arr.shape:
                result.lung_mask_available = True
                overlap = compute_lung_overlap(mask_arr, lung_arr)
                for k, v in overlap.items():
                    setattr(result, k, v)
                
                if result.outside_lung_fraction > OUTSIDE_LUNG_THRESHOLD:
                    result.add_flag("outside_lung")
    
    # SUV stats
    if suv_path.exists():
        suv_img = load_nifti(suv_path)
        if suv_img is not None:
            suv_arr = sitk.GetArrayFromImage(suv_img)
            if suv_arr.shape == mask_arr.shape:
                suv_stats = compute_suv_stats(suv_arr, mask_arr)
                for k, v in suv_stats.items():
                    setattr(result, k, v)
    
    result.status = "ok"
    return result


def compute_cohort_outliers(results: List[QCResult]) -> List[QCResult]:
    """Add cohort-based outlier flags based on z-scores."""
    # Collect HU means for cohort stats
    hu_means = [r.hu_mean for r in results if np.isfinite(r.hu_mean) and r.tumor_voxels > 0]
    
    if len(hu_means) < 10:
        return results
    
    hu_median = np.median(hu_means)
    hu_iqr = np.percentile(hu_means, 75) - np.percentile(hu_means, 25)
    hu_std = np.std(hu_means)
    
    for r in results:
        if r.tumor_voxels > 0 and np.isfinite(r.hu_mean):
            # Z-score based on cohort
            z_score = abs(r.hu_mean - np.mean(hu_means)) / hu_std if hu_std > 0 else 0
            if z_score > Z_SCORE_THRESHOLD:
                r.add_flag("hu_cohort_outlier")
            
            # IQR-based outlier
            if r.hu_mean < hu_median - 2 * hu_iqr or r.hu_mean > hu_median + 2 * hu_iqr:
                r.add_flag("hu_iqr_outlier")
    
    return results


def generate_summary_md(results: List[QCResult], output_path: Path):
    """Generate markdown summary report."""
    ok_results = [r for r in results if r.tumor_voxels > 0]
    
    # Status counts
    status_counts = {}
    for r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1
    
    # Flag counts
    flag_counts = {}
    for r in results:
        for flag in r.flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    
    # Cohort stats
    volumes = [r.tumor_vol_ml for r in ok_results]
    hu_means = [r.hu_mean for r in ok_results if np.isfinite(r.hu_mean)]
    
    with open(output_path, 'w') as f:
        f.write("# MedSAM2 Tumor Mask QC Summary\n\n")
        f.write(f"**Total patients**: {len(results)}\n\n")
        
        f.write("## Status Breakdown\n\n")
        f.write("| Status | Count |\n|--------|-------|\n")
        for status, count in sorted(status_counts.items()):
            f.write(f"| {status} | {count} |\n")
        
        f.write("\n## Flag Counts\n\n")
        f.write("| Flag | Count |\n|------|-------|\n")
        for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
            f.write(f"| {flag} | {count} |\n")
        
        if volumes:
            f.write("\n## Volume Distribution (ml)\n\n")
            f.write(f"- Min: {np.min(volumes):.2f}\n")
            f.write(f"- Median: {np.median(volumes):.2f}\n")
            f.write(f"- Max: {np.max(volumes):.2f}\n")
            f.write(f"- IQR: [{np.percentile(volumes, 25):.2f}, {np.percentile(volumes, 75):.2f}]\n")
        
        if hu_means:
            f.write("\n## HU Mean Distribution\n\n")
            f.write(f"- Median: {np.median(hu_means):.1f}\n")
            f.write(f"- Std: {np.std(hu_means):.1f}\n")
            f.write(f"- IQR: [{np.percentile(hu_means, 25):.1f}, {np.percentile(hu_means, 75):.1f}]\n")


def main():
    parser = argparse.ArgumentParser(description="QC tumor masks from MedSAM2")
    parser.add_argument("--output-dir", type=Path, default=Path("output"),
                       help="Output directory for QC reports")
    parser.add_argument("--index-file", type=Path, default=None,
                       help="Path to index_tumor_bbox_masks.csv")
    parser.add_argument("--generate-montages", action="store_true", default=True,
                       help="Generate visual montages for flagged cases")
    parser.add_argument("--montage-all", action="store_true",
                       help="Generate montages for all cases, not just flagged")
    parser.add_argument("--patients", type=str, default="",
                       help="Comma-separated patient IDs to process (default: all)")
    args = parser.parse_args()
    
    cfg = Config()
    
    # Find index file
    if args.index_file:
        index_path = args.index_file
    else:
        index_path = cfg.PROJECT_ROOT / "output" / "index_tumor_bbox_masks.csv"
    
    if not index_path.exists():
        print(f"Error: Index file not found: {index_path}")
        return 1
    
    print(f"Loading index from: {index_path}")
    df = pd.read_csv(index_path)
    print(f"Index has {len(df)} patients")

    if args.patients.strip():
        wanted = {x.strip() for x in args.patients.split(",") if x.strip()}
        df = df[df["patient_id"].astype(str).isin(wanted)].copy()
        print(f"Filtered to {len(df)} patients: {sorted(list(wanted))}")
    
    print(f"Processing {len(df)} patients")
    
    # Run QC on all patients
    results: List[QCResult] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Running QC"):
        patient_id = row["patient_id"]
        result = qc_single_patient(patient_id, row, cfg.CASES_DIR)
        results.append(result)
    
    # Add cohort outlier flags
    results = compute_cohort_outliers(results)
    
    # Generate montages for flagged cases
    flagged_results = [r for r in results if r.flags and r.tumor_voxels > 0]
    if args.montage_all:
        montage_results = [r for r in results if r.tumor_voxels > 0]
    else:
        montage_results = flagged_results
    
    if args.generate_montages and montage_results:
        print(f"\nGenerating {len(montage_results)} montages...")
        for result in tqdm(montage_results, desc="Generating montages"):
            mask_path = Path(result.mask_path)
            case_dir = mask_path.parent.parent
            output_path = case_dir / "qc" / "tumor_qc_montage.png"
            
            try:
                ct_img = load_nifti(Path(result.ct_path))
                mask_img = load_nifti(mask_path)
                if ct_img and mask_img:
                    ct_arr = sitk.GetArrayFromImage(ct_img)
                    mask_arr = sitk.GetArrayFromImage(mask_img)
                    
                    suv_arr = None
                    if result.suv_path and Path(result.suv_path).exists():
                        suv_img = load_nifti(Path(result.suv_path))
                        if suv_img:
                            suv_arr = sitk.GetArrayFromImage(suv_img)
                    
                    generate_montage(ct_arr, mask_arr, suv_arr, result, output_path)
            except Exception as e:
                print(f"Warning: Failed to generate montage for {result.patient_id}: {e}")
    
    # Save detailed results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    detailed_path = args.output_dir / "qc_tumor_masks_detailed.csv"
    detailed_records = []
    for r in results:
        record = {
            "patient_id": r.patient_id,
            "status": r.status,
            "flags": ";".join(r.flags) if r.flags else "",
            "tumor_voxels": r.tumor_voxels,
            "tumor_vol_ml": r.tumor_vol_ml,
            "n_components": r.n_components,
            "bbox_z": r.bbox_z,
            "bbox_y": r.bbox_y,
            "bbox_x": r.bbox_x,
            "centroid_z": r.centroid_z,
            "centroid_y": r.centroid_y,
            "centroid_x": r.centroid_x,
            "hu_mean": r.hu_mean,
            "hu_std": r.hu_std,
            "hu_p01": r.hu_p01,
            "hu_p50": r.hu_p50,
            "hu_p99": r.hu_p99,
            "z_span": r.z_span,
            "spikiness": r.spikiness,
            "fill_ratio_mean": r.fill_ratio_mean,
            "lung_overlap_fraction": r.lung_overlap_fraction,
            "outside_lung_fraction": r.outside_lung_fraction,
            "shape_match": r.shape_match,
            "spacing_match": r.spacing_match,
            "boxes_used": r.boxes_used,
            "missing_sop": r.missing_sop,
            "error": r.error,
        }
        detailed_records.append(record)
    
    pd.DataFrame(detailed_records).to_csv(detailed_path, index=False)
    print(f"\nSaved detailed QC to: {detailed_path}")
    
    # Save flagged cases
    flags_path = args.output_dir / "qc_flags.csv"
    flagged_records = [
        {"patient_id": r.patient_id, "flags": ";".join(r.flags), "status": r.status, 
         "tumor_vol_ml": r.tumor_vol_ml, "error": r.error}
        for r in results if r.flags
    ]
    pd.DataFrame(flagged_records).to_csv(flags_path, index=False)
    print(f"Saved {len(flagged_records)} flagged cases to: {flags_path}")
    
    # Generate summary
    summary_path = args.output_dir / "qc_tumor_masks_summary.md"
    generate_summary_md(results, summary_path)
    print(f"Saved summary to: {summary_path}")
    
    # Print quick summary
    print("\n=== QC Summary ===")
    status_counts = {}
    for r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    
    print(f"\nFlagged cases: {len(flagged_records)}")
    
    return 0


if __name__ == "__main__":
    exit(main())
