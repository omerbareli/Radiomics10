# pipeline/lidc/eval_interim.py
"""Interim evaluation: predict fold-0 validation set from checkpoint_best.pth,
compute Dice / HD95 / volume, and generate overlay PNGs.

Usage:
    python -m pipeline.lidc.eval_interim [--epoch-label 20] [--n-overlays 10]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import shutil
import subprocess
import tempfile
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
DATASET_ID = 503
DATASET_NAME = "Dataset503_LIDC_SUBSET"
LOCAL_PREP = Path("/home/asafz/projects/radiomics10/nnUNet_preprocessed_local")
LOCAL_RES = Path("/home/asafz/projects/radiomics10/nnUNet_results_local")
RAW = Path("/mnt/metlab27/D/Input/Radiomics/T_IL Radiomics10/LIDC-IDRI/Output/nnunet/nnUNet_raw")
QC_DIR = Path("/mnt/metlab27/D/Input/Radiomics/T_IL Radiomics10/LIDC-IDRI/Output/qc")
CASES_DIR = Path("/mnt/metlab27/D/Input/Radiomics/T_IL Radiomics10/LIDC-IDRI/Output/cases")


def load_splits() -> List[Dict[str, List[str]]]:
    p = LOCAL_PREP / DATASET_NAME / "splits_final.json"
    with open(p) as f:
        return json.load(f)


def predict_fold0(device: str = "cuda:0") -> Path:
    """Run nnUNetv2_predict on fold-0 val set using checkpoint_best."""
    splits = load_splits()
    val_cases = splits[0]["val"]  # 95 cases
    logger.info(f"Fold-0 validation set: {len(val_cases)} cases")

    # Temp dirs for input/output
    pred_dir = QC_DIR / "interim_predictions_fold0"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Check if predictions already exist
    existing = [f.stem for f in pred_dir.glob("*.nii.gz")]
    if set(val_cases).issubset(set(existing)):
        logger.info("All predictions already exist, skipping inference")
        return pred_dir

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Symlink val images
        images_tr = RAW / DATASET_NAME / "imagesTr"
        for cn in val_cases:
            src = images_tr / f"{cn}_0000.nii.gz"
            dst = input_dir / f"{cn}_0000.nii.gz"
            if src.exists():
                dst.symlink_to(src.resolve())
            else:
                logger.warning(f"Missing: {src}")

        env = os.environ.copy()
        env["nnUNet_raw"] = str(RAW)
        env["nnUNet_preprocessed"] = str(LOCAL_PREP)
        env["nnUNet_results"] = str(LOCAL_RES)
        # Use a different GPU to avoid contention with training
        env["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1] if ":" in device else "0"

        cmd = [
            "nnUNetv2_predict",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-d", str(DATASET_ID),
            "-c", "3d_fullres",
            "-f", "0",
            "-p", "nnUNetPlans",
            "-chk", "checkpoint_best.pth",
            "-device", "cuda",
        ]
        logger.info(f"Running inference: {' '.join(cmd)}")
        t0 = time.time()
        proc = subprocess.run(cmd, env=env, timeout=7200)
        if proc.returncode != 0:
            raise RuntimeError(f"Prediction failed with exit code {proc.returncode}")
        logger.info(f"Inference done in {time.time()-t0:.0f}s")

        # Copy predictions
        for cn in val_cases:
            src = output_dir / f"{cn}.nii.gz"
            dst = pred_dir / f"{cn}.nii.gz"
            if src.exists():
                shutil.copyfile(src, dst)

    return pred_dir


def compute_metrics(pred_dir: Path, epoch_label: str, fold: int = 0) -> Tuple[List[Dict], Dict]:
    """Compute Dice, HD95, volume for all predicted cases."""
    import SimpleITK as sitk
    from scipy.ndimage import binary_erosion, distance_transform_edt

    splits = load_splits()
    val_cases = splits[fold]["val"]

    results = []
    for cn in val_cases:
        # Map case name back to patient ID
        num = cn.replace("LIDC", "")
        patient_id = f"LIDC-IDRI-{num}"

        gt_path = CASES_DIR / patient_id / "seg" / "nodule_mask_gt.nii.gz"
        pred_path = pred_dir / f"{cn}.nii.gz"

        if not gt_path.exists() or not pred_path.exists():
            results.append({"patient_id": patient_id, "case_name": cn, "error": "missing"})
            continue

        try:
            gt_img = sitk.ReadImage(str(gt_path))
            pred_img = sitk.ReadImage(str(pred_path))
            gt = sitk.GetArrayFromImage(gt_img).astype(bool)
            pred = sitk.GetArrayFromImage(pred_img).astype(bool)
            spacing = gt_img.GetSpacing()[::-1]  # zyx

            if pred.shape != gt.shape:
                results.append({"patient_id": patient_id, "case_name": cn, "error": "shape_mismatch"})
                continue

            # Dice
            inter = np.sum(pred & gt)
            total = np.sum(pred) + np.sum(gt)
            dice = 2 * inter / total if total > 0 else (1.0 if np.sum(pred) == 0 else 0.0)

            # HD95
            hd95 = float("nan")
            if np.sum(pred) > 0 and np.sum(gt) > 0:
                ps = pred & ~binary_erosion(pred, iterations=1)
                gs = gt & ~binary_erosion(gt, iterations=1)
                if np.sum(ps) == 0: ps = pred
                if np.sum(gs) == 0: gs = gt
                d_gt = distance_transform_edt(~gs, sampling=spacing)
                d_pred = distance_transform_edt(~ps, sampling=spacing)
                hd95 = float(max(
                    np.percentile(d_gt[ps], 95),
                    np.percentile(d_pred[gs], 95),
                ))

            # Volume
            vox_vol = float(np.prod(spacing))
            gt_vol = float(np.sum(gt)) * vox_vol / 1000
            pred_vol = float(np.sum(pred)) * vox_vol / 1000

            # Precision / Recall
            tp = np.sum(pred & gt)
            fp = np.sum(pred & ~gt)
            fn = np.sum(~pred & gt)
            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

            results.append({
                "patient_id": patient_id,
                "case_name": cn,
                "dice": round(float(dice), 6),
                "hd95_mm": round(hd95, 2) if not np.isnan(hd95) else None,
                "precision": round(prec, 6),
                "recall": round(rec, 6),
                "gt_volume_ml": round(gt_vol, 4),
                "pred_volume_ml": round(pred_vol, 4),
                "volume_diff_ml": round(pred_vol - gt_vol, 4),
                "error": None,
            })
        except Exception as e:
            results.append({"patient_id": patient_id, "case_name": cn, "error": str(e)})

    # Save per-case CSV
    csv_path = QC_DIR / f"metrics_nnunet_subset_fold{fold}_epoch{epoch_label}.csv"
    fieldnames = ["patient_id", "case_name", "dice", "hd95_mm", "precision", "recall",
                  "gt_volume_ml", "pred_volume_ml", "volume_diff_ml", "error"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Per-case metrics -> {csv_path}")

    # Summary
    valid = [r for r in results if r.get("error") is None]
    if valid:
        dice_vals = [r["dice"] for r in valid]
        hd95_vals = [r["hd95_mm"] for r in valid if r["hd95_mm"] is not None]
        summary = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": DATASET_NAME,
            "fold": fold,
            "epoch_label": epoch_label,
            "n_cases": len(results),
            "n_valid": len(valid),
            "n_errors": len(results) - len(valid),
            "dice": {
                "mean": round(float(np.mean(dice_vals)), 4),
                "std": round(float(np.std(dice_vals)), 4),
                "median": round(float(np.median(dice_vals)), 4),
                "min": round(float(np.min(dice_vals)), 4),
                "max": round(float(np.max(dice_vals)), 4),
                "q25": round(float(np.percentile(dice_vals, 25)), 4),
                "q75": round(float(np.percentile(dice_vals, 75)), 4),
            },
            "hd95_mm": {
                "mean": round(float(np.mean(hd95_vals)), 2) if hd95_vals else None,
                "std": round(float(np.std(hd95_vals)), 2) if hd95_vals else None,
                "median": round(float(np.median(hd95_vals)), 2) if hd95_vals else None,
            },
        }
    else:
        summary = {"n_valid": 0, "error": "no_valid_results"}

    summary_path = QC_DIR / f"metrics_nnunet_subset_fold{fold}_epoch{epoch_label}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary -> {summary_path}")

    if valid:
        logger.info(
            f"\n{'='*60}\n"
            f"INTERIM EVALUATION: fold-{fold}, epoch ~{epoch_label}\n"
            f"{'='*60}\n"
            f"Cases: {len(valid)}/{len(results)} valid\n"
            f"Dice:  {summary['dice']['mean']:.4f} +/- {summary['dice']['std']:.4f} "
            f"(median {summary['dice']['median']:.4f})\n"
            f"HD95:  {summary['hd95_mm']['mean']:.2f} +/- {summary['hd95_mm']['std']:.2f} mm\n"
        )

    return results, summary


def generate_overlays(pred_dir: Path, results: List[Dict], n: int = 10,
                      epoch_label: str = "20", fold: int = 0,
                      overlay_mode: str = "random"):
    """Generate overlay PNGs.

    overlay_mode: "random" picks n random cases with Dice>0,
                  "topbottom" picks n best and n worst by Dice.
    """
    import SimpleITK as sitk

    overlay_dir = QC_DIR / f"overlays_fold{fold}_epoch{epoch_label}"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    valid = [r for r in results if r.get("error") is None and r.get("dice") is not None]
    if overlay_mode == "topbottom":
        sorted_by_dice = sorted(valid, key=lambda r: r["dice"])
        bottom_n = sorted_by_dice[:n]
        top_n = sorted_by_dice[-n:]
        sampled = bottom_n + [r for r in top_n if r not in bottom_n]
    else:
        valid = [r for r in valid if r.get("dice", 0) > 0]
        random.seed(42)
        sampled = random.sample(valid, min(n, len(valid)))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping overlays")
        return

    for r in sampled:
        pid = r["patient_id"]
        cn = r["case_name"]
        gt_path = CASES_DIR / pid / "seg" / "nodule_mask_gt.nii.gz"
        pred_path = pred_dir / f"{cn}.nii.gz"
        ct_path = CASES_DIR / pid / "nifti" / "ct.nii.gz"

        if not all(p.exists() for p in [gt_path, pred_path, ct_path]):
            continue

        ct_img = sitk.ReadImage(str(ct_path))
        gt_img = sitk.ReadImage(str(gt_path))
        pred_img = sitk.ReadImage(str(pred_path))

        ct = sitk.GetArrayFromImage(ct_img)
        gt = sitk.GetArrayFromImage(gt_img).astype(bool)
        pred = sitk.GetArrayFromImage(pred_img).astype(bool)

        # Find slice with max GT overlap
        gt_per_slice = gt.sum(axis=(1, 2))
        if gt_per_slice.max() == 0:
            continue
        best_z = int(np.argmax(gt_per_slice))

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, title, mask, color in [
            (axes[0], "CT", None, None),
            (axes[1], f"GT (green) + Pred (red)\nDice={r['dice']:.3f}", None, None),
            (axes[2], f"Overlay detail", None, None),
        ]:
            ax.imshow(ct[best_z], cmap="gray", vmin=-1000, vmax=400)
            ax.set_title(title, fontsize=12)
            ax.axis("off")

        # GT contour in green, pred in red
        axes[1].contour(gt[best_z], levels=[0.5], colors=["lime"], linewidths=1.5)
        axes[1].contour(pred[best_z], levels=[0.5], colors=["red"], linewidths=1.5)

        # Zoomed overlay
        gy, gx = np.where(gt[best_z])
        if len(gy) > 0:
            cy, cx = int(gy.mean()), int(gx.mean())
            margin = 60
            y0, y1 = max(0, cy - margin), min(ct.shape[1], cy + margin)
            x0, x1 = max(0, cx - margin), min(ct.shape[2], cx + margin)
            axes[2].imshow(ct[best_z, y0:y1, x0:x1], cmap="gray", vmin=-1000, vmax=400)
            axes[2].contour(gt[best_z, y0:y1, x0:x1], levels=[0.5], colors=["lime"], linewidths=2)
            axes[2].contour(pred[best_z, y0:y1, x0:x1], levels=[0.5], colors=["red"], linewidths=2)
            axes[2].set_title(f"Zoomed (z={best_z})\nHD95={r.get('hd95_mm', 'N/A')} mm", fontsize=12)

        fig.suptitle(f"{pid}  |  Dice={r['dice']:.3f}  |  GT vol={r['gt_volume_ml']:.2f} ml", fontsize=14)
        fig.tight_layout()

        out_path = overlay_dir / f"{cn}_overlay.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Overlay: {out_path.name}")

    logger.info(f"Overlays saved to {overlay_dir} ({len(sampled)} images)")


def main():
    parser = argparse.ArgumentParser(description="Interim evaluation")
    parser.add_argument("--fold", type=int, default=0, help="Fold to evaluate")
    parser.add_argument("--epoch-label", type=str, default="20", help="Label for output files")
    parser.add_argument("--n-overlays", type=int, default=10, help="Number of overlay PNGs")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU for inference")
    parser.add_argument("--skip-predict", action="store_true", help="Skip prediction, just compute metrics")
    parser.add_argument("--pred-dir", type=str, default=None,
                        help="Path to prediction dir (overrides default)")
    parser.add_argument("--overlay-mode", type=str, default="random",
                        choices=["random", "topbottom"],
                        help="Overlay selection: random or topbottom (best+worst)")
    args = parser.parse_args()

    if args.pred_dir:
        pred_dir = Path(args.pred_dir)
    elif not args.skip_predict:
        pred_dir = predict_fold0(device=args.device)
    else:
        suffix = f"_epoch{args.epoch_label}" if args.epoch_label else ""
        pred_dir = QC_DIR / f"interim_predictions_fold{args.fold}{suffix}"

    results, summary = compute_metrics(pred_dir, args.epoch_label, fold=args.fold)
    generate_overlays(pred_dir, results, n=args.n_overlays,
                      epoch_label=args.epoch_label, fold=args.fold,
                      overlay_mode=args.overlay_mode)

    logger.info("Interim evaluation complete!")


if __name__ == "__main__":
    main()
