# pipeline/lidc/evaluate_heldout.py
"""Evaluate ensemble predictions on held-out OK cases (not used in training).

Reads predictions from the ensemble output directory, computes metrics
against ground truth, and generates overlay PNGs.

Usage:
    python -m pipeline.lidc.evaluate_heldout
    python -m pipeline.lidc.evaluate_heldout --copy-to-cases
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
MOUNT_ROOT = Path(
    "/mnt/metlab27/D/Input/Radiomics/T_IL Radiomics10/LIDC-IDRI"
)
OUTPUT_ROOT = MOUNT_ROOT / "Output"
QC_DIR = OUTPUT_ROOT / "qc"
CASES_DIR = OUTPUT_ROOT / "cases"

# Default prediction directory (4-fold ensemble)
DEFAULT_PRED_DIR = OUTPUT_ROOT / "nnunet" / "heldout_predictions_4fold"

# Held-out case list
HELDOUT_IDS_FILE = QC_DIR / "heldout_case_ids.txt"


def case_to_patient_id(case_name: str) -> str:
    """LIDC0001 -> LIDC-IDRI-0001"""
    num = case_name.replace("LIDC", "")
    return f"LIDC-IDRI-{num}"


def patient_id_to_case_name(patient_id: str) -> str:
    """LIDC-IDRI-0001 -> LIDC0001"""
    num = patient_id.split("-")[-1]
    return f"LIDC{num}"


def compute_case_metrics(pred_path: Path, gt_path: Path) -> Dict[str, Any]:
    """Compute Dice, HD95, precision, recall, volume for one case."""
    import SimpleITK as sitk
    from scipy.ndimage import binary_erosion, distance_transform_edt

    gt_img = sitk.ReadImage(str(gt_path))
    pred_img = sitk.ReadImage(str(pred_path))
    gt = sitk.GetArrayFromImage(gt_img).astype(bool)
    pred = sitk.GetArrayFromImage(pred_img).astype(bool)
    spacing = gt_img.GetSpacing()[::-1]  # zyx

    if pred.shape != gt.shape:
        return {"error": "shape_mismatch"}

    inter = int(np.sum(pred & gt))
    total = int(np.sum(pred)) + int(np.sum(gt))
    dice = 2 * inter / total if total > 0 else (1.0 if np.sum(pred) == 0 else 0.0)

    hd95 = float("nan")
    if np.sum(pred) > 0 and np.sum(gt) > 0:
        ps = pred & ~binary_erosion(pred, iterations=1)
        gs = gt & ~binary_erosion(gt, iterations=1)
        if np.sum(ps) == 0:
            ps = pred
        if np.sum(gs) == 0:
            gs = gt
        d_gt = distance_transform_edt(~gs, sampling=spacing)
        d_pred = distance_transform_edt(~ps, sampling=spacing)
        hd95 = float(
            max(np.percentile(d_gt[ps], 95), np.percentile(d_pred[gs], 95))
        )

    vox_vol = float(np.prod(spacing))
    gt_vol = float(np.sum(gt)) * vox_vol / 1000
    pred_vol = float(np.sum(pred)) * vox_vol / 1000

    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))

    return {
        "dice": round(float(dice), 6),
        "hd95_mm": round(hd95, 2) if not np.isnan(hd95) else None,
        "precision": round(float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0, 6),
        "recall": round(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0, 6),
        "gt_volume_ml": round(gt_vol, 4),
        "pred_volume_ml": round(pred_vol, 4),
        "volume_diff_ml": round(pred_vol - gt_vol, 4),
        "n_ref": int(np.sum(gt)),
        "n_pred": int(np.sum(pred)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "error": None,
    }


def compute_all_metrics(
    pred_dir: Path, heldout_ids: List[str]
) -> Tuple[List[Dict], Dict]:
    """Compute metrics for all held-out cases."""
    results = []
    t0 = time.time()

    for i, pid in enumerate(heldout_ids):
        cn = patient_id_to_case_name(pid)
        pred_path = pred_dir / f"{cn}.nii.gz"
        gt_path = CASES_DIR / pid / "seg" / "nodule_mask_gt.nii.gz"

        row = {"case_name": cn, "patient_id": pid}

        if not pred_path.exists():
            row["error"] = "missing_prediction"
            results.append(row)
            continue
        if not gt_path.exists():
            row["error"] = "missing_gt"
            results.append(row)
            continue

        try:
            metrics = compute_case_metrics(pred_path, gt_path)
            row.update(metrics)
        except Exception as e:
            row["error"] = str(e)

        results.append(row)
        if (i + 1) % 50 == 0:
            logger.info(f"  Metrics: {i+1}/{len(heldout_ids)} ({time.time()-t0:.0f}s)")

    elapsed = time.time() - t0
    n_valid = sum(1 for r in results if r.get("error") is None)
    logger.info(f"Metrics computed for {n_valid} cases in {elapsed:.0f}s")
    return results, _build_summary(results)


def _build_summary(results: List[Dict]) -> Dict:
    valid = [r for r in results if r.get("error") is None]
    if not valid:
        return {"n_valid": 0, "error": "no_valid_results"}

    dice_vals = [r["dice"] for r in valid]
    hd95_vals = [r["hd95_mm"] for r in valid if r["hd95_mm"] is not None]

    # Failure analysis
    zero_dice = [r for r in valid if r["dice"] == 0.0]
    zero_analysis = []
    for r in zero_dice:
        if r["n_ref"] == 0:
            ftype = "missing_gt"
        elif r["n_pred"] == 0:
            ftype = "complete_miss"
        else:
            ftype = "fp_only_no_overlap"
        zero_analysis.append({
            "case_name": r["case_name"],
            "patient_id": r["patient_id"],
            "n_ref": r["n_ref"],
            "n_pred": r["n_pred"],
            "failure_type": ftype,
        })

    # Size-stratified
    gt_vols = [r["gt_volume_ml"] for r in valid]
    q25, q50, q75 = np.percentile(gt_vols, [25, 50, 75])
    size_strata = {}
    for label, lo, hi in [
        ("Q1_smallest", 0, q25),
        ("Q2", q25, q50),
        ("Q3", q50, q75),
        ("Q4_largest", q75, float("inf")),
    ]:
        stratum = [r for r in valid if lo <= r["gt_volume_ml"] < hi]
        if stratum:
            sd = [r["dice"] for r in stratum]
            size_strata[label] = {
                "n": len(stratum),
                "gt_vol_range_ml": f"{lo:.2f}-{hi:.2f}",
                "dice_mean": round(float(np.mean(sd)), 4),
                "dice_median": round(float(np.median(sd)), 4),
            }

    nonzero_dice = [r["dice"] for r in valid if r["dice"] > 0]

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "heldout_ensemble",
        "n_cases_total": len(results),
        "n_valid": len(valid),
        "n_errors": len(results) - len(valid),
        "n_missing_predictions": sum(
            1 for r in results if r.get("error") == "missing_prediction"
        ),
        "dice": {
            "mean": round(float(np.mean(dice_vals)), 4),
            "std": round(float(np.std(dice_vals)), 4),
            "median": round(float(np.median(dice_vals)), 4),
            "min": round(float(np.min(dice_vals)), 4),
            "max": round(float(np.max(dice_vals)), 4),
            "p5": round(float(np.percentile(dice_vals, 5)), 4),
            "p10": round(float(np.percentile(dice_vals, 10)), 4),
            "q25": round(float(np.percentile(dice_vals, 25)), 4),
            "q75": round(float(np.percentile(dice_vals, 75)), 4),
            "p90": round(float(np.percentile(dice_vals, 90)), 4),
            "p95": round(float(np.percentile(dice_vals, 95)), 4),
        },
        "dice_excluding_zero": {
            "n": len(nonzero_dice),
            "mean": round(float(np.mean(nonzero_dice)), 4) if nonzero_dice else None,
            "median": round(float(np.median(nonzero_dice)), 4) if nonzero_dice else None,
        },
        "hd95_mm": {
            "mean": round(float(np.mean(hd95_vals)), 2) if hd95_vals else None,
            "std": round(float(np.std(hd95_vals)), 2) if hd95_vals else None,
            "median": round(float(np.median(hd95_vals)), 2) if hd95_vals else None,
        },
        "failure_analysis": {
            "n_dice_zero": len(zero_dice),
            "pct_dice_zero": round(100 * len(zero_dice) / len(valid), 1) if valid else 0,
            "cases": zero_analysis,
        },
        "size_stratified_dice": size_strata,
    }


def generate_overlays(
    results: List[Dict],
    pred_dir: Path,
    output_dir: Path,
    n_random: int = 10,
    n_topbottom: int = 5,
    overlay_mode: str = "topbottom",
) -> None:
    """Generate overlay PNGs for selected cases."""
    import SimpleITK as sitk

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping overlays")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    valid = [r for r in results if r.get("error") is None and r.get("dice") is not None]
    sampled = []

    if overlay_mode == "topbottom":
        sorted_by_dice = sorted(valid, key=lambda r: r["dice"])
        sampled = sorted_by_dice[:n_topbottom] + sorted_by_dice[-n_topbottom:]
        zero_extra = [r for r in valid if r["dice"] == 0.0 and r not in sampled]
        sampled.extend(zero_extra)
        # Add random
        random.seed(42)
        candidates = [r for r in valid if r not in sampled and r["dice"] > 0]
        sampled.extend(random.sample(candidates, min(n_random, len(candidates))))
    else:
        candidates = [r for r in valid if r.get("dice", 0) > 0]
        random.seed(42)
        sampled = random.sample(candidates, min(n_random, len(candidates)))

    logger.info(f"Generating {len(sampled)} overlays...")

    for idx, r in enumerate(sampled):
        pid = r["patient_id"]
        cn = r["case_name"]
        pred_path = pred_dir / f"{cn}.nii.gz"
        gt_path = CASES_DIR / pid / "seg" / "nodule_mask_gt.nii.gz"
        ct_path = CASES_DIR / pid / "nifti" / "ct.nii.gz"

        if not all(p.exists() for p in [pred_path, gt_path, ct_path]):
            continue

        ct_img = sitk.ReadImage(str(ct_path))
        gt_img = sitk.ReadImage(str(gt_path))
        pred_img = sitk.ReadImage(str(pred_path))
        ct = sitk.GetArrayFromImage(ct_img)
        gt = sitk.GetArrayFromImage(gt_img).astype(bool)
        pred = sitk.GetArrayFromImage(pred_img).astype(bool)

        combined = gt.sum(axis=(1, 2)) + pred.sum(axis=(1, 2))
        if combined.max() == 0:
            continue
        best_z = int(np.argmax(combined))

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(ct[best_z], cmap="gray", vmin=-1000, vmax=400)
        axes[0].set_title("CT", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(ct[best_z], cmap="gray", vmin=-1000, vmax=400)
        if gt[best_z].any():
            axes[1].contour(gt[best_z], levels=[0.5], colors=["lime"], linewidths=1.5)
        if pred[best_z].any():
            axes[1].contour(pred[best_z], levels=[0.5], colors=["red"], linewidths=1.5)
        axes[1].set_title(f"GT (green) + Pred (red)\nDice={r['dice']:.3f}", fontsize=12)
        axes[1].axis("off")

        gy, gx = np.where(gt[best_z] | pred[best_z])
        if len(gy) > 0:
            cy, cx = int(gy.mean()), int(gx.mean())
            margin = 60
            y0, y1 = max(0, cy - margin), min(ct.shape[1], cy + margin)
            x0, x1 = max(0, cx - margin), min(ct.shape[2], cx + margin)
            axes[2].imshow(ct[best_z, y0:y1, x0:x1], cmap="gray", vmin=-1000, vmax=400)
            if gt[best_z, y0:y1, x0:x1].any():
                axes[2].contour(gt[best_z, y0:y1, x0:x1], levels=[0.5], colors=["lime"], linewidths=2)
            if pred[best_z, y0:y1, x0:x1].any():
                axes[2].contour(pred[best_z, y0:y1, x0:x1], levels=[0.5], colors=["red"], linewidths=2)
            hd_str = f"{r['hd95_mm']:.1f}" if r.get("hd95_mm") else "N/A"
            axes[2].set_title(f"Zoomed (z={best_z})\nHD95={hd_str} mm", fontsize=12)
        axes[2].axis("off")

        fig.suptitle(
            f"{pid} (held-out)  |  Dice={r['dice']:.3f}  |  GT vol={r['gt_volume_ml']:.2f} ml",
            fontsize=14,
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"{cn}_overlay.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        if (idx + 1) % 10 == 0:
            logger.info(f"  Overlays: {idx+1}/{len(sampled)}")

    logger.info(f"Overlays saved to {output_dir} ({len(sampled)} images)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ensemble predictions on held-out OK cases"
    )
    parser.add_argument(
        "--pred-dir", type=str, default=str(DEFAULT_PRED_DIR),
        help="Directory with ensemble prediction NIfTIs",
    )
    parser.add_argument(
        "--n-overlays", type=int, default=10,
        help="Number of random overlay PNGs (default: 10)",
    )
    parser.add_argument(
        "--overlay-mode", type=str, default="topbottom",
        choices=["random", "topbottom"],
    )
    parser.add_argument(
        "--output-prefix", type=str, default="metrics_nnunet_heldout_4fold",
    )
    parser.add_argument(
        "--copy-to-cases", action="store_true",
        help="Copy ensemble predictions to per-case seg directories",
    )
    parser.add_argument(
        "--skip-overlays", action="store_true",
    )
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    if not pred_dir.exists():
        logger.error(f"Prediction directory not found: {pred_dir}")
        logger.error("Run ensemble inference first (nnUNetv2_predict)")
        return

    # Load held-out case list
    if not HELDOUT_IDS_FILE.exists():
        logger.error(f"Held-out case list not found: {HELDOUT_IDS_FILE}")
        logger.error("Run prepare_heldout_input.py first")
        return

    with open(HELDOUT_IDS_FILE) as f:
        heldout_ids = [line.strip() for line in f if line.strip()]
    logger.info(f"Held-out cases: {len(heldout_ids)}")

    # Compute metrics
    results, summary = compute_all_metrics(pred_dir, heldout_ids)

    # Save CSV
    csv_path = QC_DIR / f"{args.output_prefix}.csv"
    fieldnames = [
        "case_name", "patient_id", "dice", "hd95_mm",
        "precision", "recall", "gt_volume_ml", "pred_volume_ml",
        "volume_diff_ml", "n_ref", "n_pred", "error",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Per-case CSV -> {csv_path}")

    # Save summary
    summary_path = QC_DIR / f"{args.output_prefix}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary -> {summary_path}")

    # Print summary
    valid = [r for r in results if r.get("error") is None]
    if valid:
        d = summary["dice"]
        logger.info(
            f"\n{'='*60}\n"
            f"HELD-OUT EVALUATION ({len(valid)} cases)\n"
            f"{'='*60}\n"
            f"Cases:  {summary['n_valid']}/{summary['n_cases_total']} valid\n"
            f"Missing predictions: {summary['n_missing_predictions']}\n"
            f"Dice:   {d['mean']:.4f} +/- {d['std']:.4f} "
            f"(median {d['median']:.4f})\n"
            f"HD95:   {summary['hd95_mm']['mean']} +/- {summary['hd95_mm']['std']} mm\n"
            f"Dice=0: {summary['failure_analysis']['n_dice_zero']} cases "
            f"({summary['failure_analysis']['pct_dice_zero']:.1f}%)\n"
        )

    # Overlays
    if not args.skip_overlays:
        overlay_dir = QC_DIR / f"overlays_{args.output_prefix}"
        generate_overlays(
            results, pred_dir, overlay_dir,
            n_random=args.n_overlays,
            overlay_mode=args.overlay_mode,
        )

    # Copy to case directories
    if args.copy_to_cases:
        copied = 0
        for r in results:
            if r.get("error"):
                continue
            cn = r["case_name"]
            pid = r["patient_id"]
            src = pred_dir / f"{cn}.nii.gz"
            dst = CASES_DIR / pid / "seg" / "nodule_mask_nnunet.nii.gz"
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                copied += 1
        logger.info(f"Copied {copied} predictions to case directories")

    logger.info("Held-out evaluation complete!")


if __name__ == "__main__":
    main()
