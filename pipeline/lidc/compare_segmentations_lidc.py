# pipeline/lidc/compare_segmentations_lidc.py
"""Compare nnU-Net OOF predictions against ground truth masks.

Computes metrics:
- Dice coefficient
- 95th percentile Hausdorff distance (HD95)
- Volume difference (ml)
- Precision (PPV)
- Recall (Sensitivity)

Usage:
    python -m pipeline.lidc.compare_segmentations_lidc --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from pipeline.lidc.config import LIDCConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice coefficient."""
    intersection = np.sum(pred & gt)
    total = np.sum(pred) + np.sum(gt)
    if total == 0:
        return 1.0 if np.sum(pred) == 0 else 0.0
    return 2 * intersection / total


def compute_hd95(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: Tuple[float, float, float],
) -> float:
    """Compute 95th percentile Hausdorff distance in mm."""
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return float("nan")
    
    # Get surface voxels
    pred_surface = pred & ~ndi_erode(pred)
    gt_surface = gt & ~ndi_erode(gt)
    
    if np.sum(pred_surface) == 0 or np.sum(gt_surface) == 0:
        # Fall back to all foreground voxels
        pred_surface = pred
        gt_surface = gt
    
    # Distance transform from GT surface
    dist_from_gt = distance_transform_edt(~gt_surface, sampling=spacing)
    dist_from_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    
    # Hausdorff distances
    hd_pred_to_gt = dist_from_gt[pred_surface]
    hd_gt_to_pred = dist_from_pred[gt_surface]
    
    if len(hd_pred_to_gt) == 0 or len(hd_gt_to_pred) == 0:
        return float("nan")
    
    hd95 = max(
        np.percentile(hd_pred_to_gt, 95),
        np.percentile(hd_gt_to_pred, 95),
    )
    return float(hd95)


def ndi_erode(arr: np.ndarray) -> np.ndarray:
    """Simple binary erosion."""
    from scipy.ndimage import binary_erosion
    return binary_erosion(arr, iterations=1)


def compute_volume_ml(
    arr: np.ndarray,
    spacing: Tuple[float, float, float],
) -> float:
    """Compute volume in milliliters (cm³)."""
    voxel_vol_mm3 = np.prod(spacing)
    volume_mm3 = np.sum(arr) * voxel_vol_mm3
    volume_ml = volume_mm3 / 1000.0  # mm³ -> ml
    return float(volume_ml)


def compute_precision_recall(
    pred: np.ndarray, gt: np.ndarray
) -> Tuple[float, float]:
    """Compute precision (PPV) and recall (sensitivity)."""
    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return float(precision), float(recall)


def load_mask(path: Path) -> Tuple[np.ndarray, Tuple[float, ...]]:
    """Load mask and return (array, spacing)."""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img) > 0
    spacing = img.GetSpacing()[::-1]  # zyx order
    return arr.astype(bool), tuple(spacing)


def compute_metrics_for_case(
    patient_id: str,
    cfg: LIDCConfig,
) -> Optional[Dict[str, Any]]:
    """Compute all metrics for a single case."""
    case_dir = cfg.CASES_DIR / patient_id / "seg"
    gt_path = case_dir / "nodule_mask_gt.nii.gz"
    pred_path = case_dir / "nodule_mask_nnunet_oof.nii.gz"
    
    if not gt_path.exists():
        return {"patient_id": patient_id, "error": "GT not found"}
    if not pred_path.exists():
        return {"patient_id": patient_id, "error": "Pred not found"}
    
    try:
        gt, spacing = load_mask(gt_path)
        pred, _ = load_mask(pred_path)
        
        # Ensure same shape
        if pred.shape != gt.shape:
            return {
                "patient_id": patient_id,
                "error": f"Shape mismatch: pred={pred.shape}, gt={gt.shape}",
            }
        
        dice = compute_dice(pred, gt)
        hd95 = compute_hd95(pred, gt, spacing)
        precision, recall = compute_precision_recall(pred, gt)
        
        gt_vol = compute_volume_ml(gt, spacing)
        pred_vol = compute_volume_ml(pred, spacing)
        vol_diff = pred_vol - gt_vol
        vol_diff_pct = (vol_diff / gt_vol * 100) if gt_vol > 0 else float("nan")
        
        return {
            "patient_id": patient_id,
            "dice": dice,
            "hd95_mm": hd95,
            "precision": precision,
            "recall": recall,
            "gt_volume_ml": gt_vol,
            "pred_volume_ml": pred_vol,
            "volume_diff_ml": vol_diff,
            "volume_diff_pct": vol_diff_pct,
            "error": None,
        }
    except Exception as e:
        return {"patient_id": patient_id, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Compare nnU-Net OOF predictions against GT"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for OOF predictions (default: 42)",
    )
    args = parser.parse_args()
    
    cfg = LIDCConfig()
    
    # Load OOF mapping to get all patient IDs with predictions
    mapping_path = cfg.QC_DIR / "nnunet_oof_mapping.csv"
    if not mapping_path.exists():
        logger.error(f"OOF mapping not found: {mapping_path}")
        logger.info("Run run_nnunet_oof_predict.py first")
        return
    
    with open(mapping_path) as f:
        reader = csv.DictReader(f)
        mapping = list(reader)
    
    patient_ids = [row["patient_id"] for row in mapping]
    logger.info(f"Found {len(patient_ids)} cases with OOF predictions")
    
    # Compute metrics for each case
    results: List[Dict[str, Any]] = []
    errors = 0
    
    for patient_id in tqdm(patient_ids, desc="Computing metrics"):
        metrics = compute_metrics_for_case(patient_id, cfg)
        results.append(metrics)
        if metrics.get("error"):
            errors += 1
    
    # Save detailed results CSV
    csv_path = cfg.QC_DIR / "metrics_nnunet_oof_vs_gt.csv"
    fieldnames = [
        "patient_id",
        "dice",
        "hd95_mm",
        "precision",
        "recall",
        "gt_volume_ml",
        "pred_volume_ml",
        "volume_diff_ml",
        "volume_diff_pct",
        "error",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"✓ Metrics CSV: {csv_path}")
    
    # Compute summary statistics
    valid_results = [r for r in results if r.get("error") is None]
    
    if valid_results:
        dice_vals = [r["dice"] for r in valid_results]
        hd95_vals = [r["hd95_mm"] for r in valid_results if not np.isnan(r["hd95_mm"])]
        prec_vals = [r["precision"] for r in valid_results]
        recall_vals = [r["recall"] for r in valid_results]
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "seed": args.seed,
            "n_cases_total": len(patient_ids),
            "n_cases_valid": len(valid_results),
            "n_errors": errors,
            "dice": {
                "mean": float(np.mean(dice_vals)),
                "std": float(np.std(dice_vals)),
                "median": float(np.median(dice_vals)),
                "min": float(np.min(dice_vals)),
                "max": float(np.max(dice_vals)),
            },
            "hd95_mm": {
                "mean": float(np.mean(hd95_vals)) if hd95_vals else None,
                "std": float(np.std(hd95_vals)) if hd95_vals else None,
                "median": float(np.median(hd95_vals)) if hd95_vals else None,
            },
            "precision": {
                "mean": float(np.mean(prec_vals)),
                "std": float(np.std(prec_vals)),
            },
            "recall": {
                "mean": float(np.mean(recall_vals)),
                "std": float(np.std(recall_vals)),
            },
        }
    else:
        summary = {
            "timestamp": datetime.now().isoformat(),
            "seed": args.seed,
            "n_cases_total": len(patient_ids),
            "n_cases_valid": 0,
            "n_errors": errors,
            "error": "No valid results",
        }
    
    # Save summary JSON
    summary_path = cfg.QC_DIR / f"metrics_nnunet_oof_summary_seed{args.seed}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✓ Summary JSON: {summary_path}")
    
    # Print summary
    logger.info(
        f"\n{'='*50}\n"
        f"EVALUATION COMPLETE\n"
        f"{'='*50}\n"
        f"Cases: {len(valid_results)}/{len(patient_ids)} valid\n"
        f"Errors: {errors}\n"
    )
    
    if valid_results:
        logger.info(
            f"Dice:      {summary['dice']['mean']:.4f} ± {summary['dice']['std']:.4f} "
            f"(median: {summary['dice']['median']:.4f})\n"
            f"HD95 (mm): {summary['hd95_mm']['mean']:.2f} ± {summary['hd95_mm']['std']:.2f}\n"
            f"Precision: {summary['precision']['mean']:.4f} ± {summary['precision']['std']:.4f}\n"
            f"Recall:    {summary['recall']['mean']:.4f} ± {summary['recall']['std']:.4f}\n"
        )


if __name__ == "__main__":
    main()
