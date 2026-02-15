# pipeline/lidc/train_nnunet_subset.py
"""Train nnU-Net on a deterministic subset of LIDC-IDRI that fits local disk.

Workflow:
  1. Measure per-case footprint from existing preprocessed data on mount
  2. Compute maximum N that fits in local disk budget
  3. Select N cases deterministically (seed=42)
  4. Create Dataset503_LIDC_SUBSET (raw symlinks on mount, preprocessed+results local)
  5. Run nnUNetv2_plan_and_preprocess locally
  6. Train fold-0 (optionally all 5 folds)
  7. Generate OOF predictions and compute metrics

Usage:
    # Full pipeline:
    python -m pipeline.lidc.train_nnunet_subset

    # Individual steps:
    python -m pipeline.lidc.train_nnunet_subset --step select
    python -m pipeline.lidc.train_nnunet_subset --step create_dataset
    python -m pipeline.lidc.train_nnunet_subset --step preprocess
    python -m pipeline.lidc.train_nnunet_subset --step train --folds 0
    python -m pipeline.lidc.train_nnunet_subset --step predict --folds 0
    python -m pipeline.lidc.train_nnunet_subset --step evaluate
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
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.lidc.config import LIDCConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
SUBSET_DATASET_ID = 503
SUBSET_DATASET_NAME = "Dataset503_LIDC_SUBSET"
FULL_DATASET_NAME = "Dataset502_LIDC"
SEED = 42
UNPACK_FACTOR = 2.2          # npz → npy expansion factor (conservative)
SAFETY_MARGIN_GB = 40        # reserved for system + headroom
RESULTS_OVERHEAD_GB = 5      # checkpoints + logs per fold
LOCAL_PREPROCESSED_ROOT = Path("/home/asafz/projects/radiomics10/nnUNet_preprocessed_local")
LOCAL_RESULTS_ROOT = Path("/home/asafz/projects/radiomics10/nnUNet_results_local")


def get_cfg() -> LIDCConfig:
    return LIDCConfig()


def get_available_disk_gb(path: Path = Path("/")) -> float:
    """Return available disk space in GB for the filesystem containing path."""
    stat = os.statvfs(str(path))
    return (stat.f_bavail * stat.f_frsize) / (1024 ** 3)


def get_used_disk_gb(path: Path) -> float:
    """Return disk usage of a directory tree in GB."""
    if not path.exists():
        return 0.0
    total = 0
    for f in path.rglob("*"):
        if f.is_file() and not f.is_symlink():
            total += f.stat().st_size
    return total / (1024 ** 3)


# ── Step 1: Measure per-case footprint ─────────────────────────────────────
def measure_per_case_footprint(cfg: LIDCConfig) -> Dict[str, Any]:
    """Measure per-case preprocessed file sizes from mount."""
    prep_dir = (
        cfg.NNUNET_PREPROCESSED / FULL_DATASET_NAME / "nnUNetPlans_3d_fullres"
    )
    if not prep_dir.exists():
        raise FileNotFoundError(f"Preprocessed dir not found: {prep_dir}")

    npz_sizes = []
    pkl_sizes = []
    case_ids = []
    for f in sorted(prep_dir.iterdir()):
        sz = f.stat().st_size
        if f.suffix == ".npz":
            npz_sizes.append(sz)
            case_ids.append(f.stem)
        elif f.suffix == ".pkl":
            pkl_sizes.append(sz)

    n = len(npz_sizes)
    avg_npz = sum(npz_sizes) / n
    avg_pkl = sum(pkl_sizes) / n if pkl_sizes else 0
    p95_npz = sorted(npz_sizes)[int(n * 0.95)]

    stats = {
        "n_cases_preprocessed": n,
        "total_npz_gb": sum(npz_sizes) / 1e9,
        "avg_npz_mb": avg_npz / 1e6,
        "median_npz_mb": sorted(npz_sizes)[n // 2] / 1e6,
        "p95_npz_mb": p95_npz / 1e6,
        "max_npz_mb": max(npz_sizes) / 1e6,
        "avg_pkl_kb": avg_pkl / 1e3,
        "avg_total_per_case_mb": (avg_npz + avg_pkl) / 1e6,
        "case_ids": case_ids,
    }
    logger.info(
        f"Footprint: {n} cases, avg NPZ={stats['avg_npz_mb']:.1f} MB, "
        f"P95={stats['p95_npz_mb']:.1f} MB"
    )
    return stats


# ── Step 2: Compute N ──────────────────────────────────────────────────────
def compute_subset_size(stats: Dict[str, Any], n_folds_to_train: int = 1) -> Dict[str, Any]:
    """Compute the maximum subset size N that fits local disk."""
    available_gb = get_available_disk_gb(Path("/"))
    budget_gb = available_gb - SAFETY_MARGIN_GB - RESULTS_OVERHEAD_GB * n_folds_to_train

    avg_npz_mb = stats["avg_npz_mb"]
    # Disk per case = npz (stays) + npy (unpacked during training)
    per_case_mb = avg_npz_mb * (1 + UNPACK_FACTOR)
    budget_mb = budget_gb * 1000

    N = int(budget_mb / per_case_mb)
    # Clamp to available preprocessed cases
    N = min(N, stats["n_cases_preprocessed"])

    calc = {
        "available_gb": round(available_gb, 1),
        "safety_margin_gb": SAFETY_MARGIN_GB,
        "results_overhead_gb": RESULTS_OVERHEAD_GB * n_folds_to_train,
        "budget_for_preprocessed_gb": round(budget_gb, 1),
        "avg_npz_mb": round(avg_npz_mb, 2),
        "unpack_factor": UNPACK_FACTOR,
        "per_case_total_mb": round(per_case_mb, 2),
        "max_N": N,
        "projected_preprocessed_gb": round(N * per_case_mb / 1000, 1),
        "headroom_gb": round(available_gb - N * per_case_mb / 1000 - RESULTS_OVERHEAD_GB * n_folds_to_train, 1),
    }
    logger.info(
        f"Budget: {available_gb:.0f} GB avail, {budget_gb:.0f} GB for data "
        f"-> N={N} cases ({N * per_case_mb / 1000:.0f} GB projected)"
    )
    return calc


# ── Step 3: Select subset deterministically ────────────────────────────────
def select_subset(
    cfg: LIDCConfig,
    N: int,
    stats: Dict[str, Any],
    calc: Dict[str, Any],
) -> List[str]:
    """Select N cases deterministically from OK cases."""
    # Get OK cases from QC report
    qc_path = cfg.QC_DIR / "lidc_full_qc_report.csv"
    df = pd.read_csv(qc_path)
    ok_df = df[df["qc_flag"] == "OK"]

    # Filter to cases that actually have preprocessed data
    preprocessed_case_ids = set(stats["case_ids"])
    ok_patient_ids = sorted(ok_df["patient_id"].tolist())
    # Map patient IDs to case names used in preprocessing
    available = []
    for pid in ok_patient_ids:
        case_name = pid.replace("LIDC-IDRI-", "LIDC")
        if case_name in preprocessed_case_ids:
            available.append(pid)

    logger.info(f"Available OK cases with preprocessing: {len(available)}")

    # Deterministic selection
    random.seed(SEED)
    shuffled = sorted(available)
    random.shuffle(shuffled)
    selected = sorted(shuffled[:N])

    # Save case IDs
    ids_path = cfg.QC_DIR / "nnunet_subset_case_ids_seed42.txt"
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.write_text("\n".join(selected) + "\n")
    logger.info(f"Saved {len(selected)} case IDs -> {ids_path}")

    # Save selection metadata
    meta = {
        "N": len(selected),
        "seed": SEED,
        "method": "random_sample_from_ok_cases_with_preprocessing",
        "total_ok_cases": len(ok_patient_ids),
        "total_with_preprocessing": len(available),
        "dataset_id": SUBSET_DATASET_ID,
        "dataset_name": SUBSET_DATASET_NAME,
        "disk_budget": calc,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = cfg.QC_DIR / "nnunet_subset_selection.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved selection metadata -> {meta_path}")

    return selected


# ── Step 4: Create raw dataset (symlinks) ──────────────────────────────────
def create_subset_dataset(cfg: LIDCConfig, selected_ids: List[str]) -> Path:
    """Create Dataset503_LIDC_SUBSET with symlinks in nnUNet_raw on the mount."""
    # Raw dataset lives on the mount (as requested)
    dataset_path = cfg.NNUNET_RAW / SUBSET_DATASET_NAME
    images_tr = dataset_path / "imagesTr"
    labels_tr = dataset_path / "labelsTr"

    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating {SUBSET_DATASET_NAME} at {dataset_path}")

    linked = 0
    for patient_id in selected_ids:
        case_name = patient_id.replace("LIDC-IDRI-", "LIDC")

        # Source: original Dataset502 raw or case directory
        # Try case directory first (where the actual niftis live)
        ct_src = cfg.CASES_DIR / patient_id / "nifti" / "ct.nii.gz"
        mask_src = cfg.CASES_DIR / patient_id / "seg" / "nodule_mask_gt.nii.gz"

        if not ct_src.exists() or not mask_src.exists():
            logger.warning(f"[{patient_id}] Missing CT or mask, skipping")
            continue

        img_dst = images_tr / f"{case_name}_0000.nii.gz"
        lbl_dst = labels_tr / f"{case_name}.nii.gz"

        # Create symlinks (or copy if symlinks fail on CIFS)
        for src, dst in [(ct_src, img_dst), (mask_src, lbl_dst)]:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            try:
                dst.symlink_to(src)
            except OSError:
                shutil.copyfile(src, dst)

        linked += 1

    # dataset.json
    dataset_json = {
        "dataset_name": SUBSET_DATASET_NAME,
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "nodule": 1},
        "numTraining": linked,
        "file_ending": ".nii.gz",
    }
    with open(dataset_path / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    logger.info(f"Created dataset: {linked}/{len(selected_ids)} cases linked")
    return dataset_path


# ── Step 5: Generate splits ───────────────────────────────────────────────
def generate_subset_splits(
    selected_ids: List[str], n_folds: int = 5
) -> List[Dict[str, List[str]]]:
    """Generate reproducible k-fold splits for the subset."""
    random.seed(SEED)
    ids = sorted(selected_ids)
    random.shuffle(ids)

    fold_size = len(ids) // n_folds
    remainder = len(ids) % n_folds

    folds = []
    start = 0
    for i in range(n_folds):
        size = fold_size + (1 if i < remainder else 0)
        folds.append(ids[start : start + size])
        start += size

    splits = []
    for i in range(n_folds):
        val_ids = folds[i]
        train_ids = [pid for j in range(n_folds) if j != i for pid in folds[j]]
        val_names = sorted(pid.replace("LIDC-IDRI-", "LIDC") for pid in val_ids)
        train_names = sorted(pid.replace("LIDC-IDRI-", "LIDC") for pid in train_ids)
        splits.append({"train": train_names, "val": val_names})

    # Verify no leakage
    all_val = set()
    for i, s in enumerate(splits):
        overlap = set(s["train"]) & set(s["val"])
        if overlap:
            raise RuntimeError(f"Leakage in fold {i}: {overlap}")
        all_val.update(s["val"])
    all_names = set(pid.replace("LIDC-IDRI-", "LIDC") for pid in selected_ids)
    if all_val != all_names:
        raise RuntimeError("Split union != selected cases")

    logger.info(f"Generated {n_folds}-fold splits, leakage-free verified")
    return splits


def save_subset_splits(
    splits: List[Dict[str, List[str]]], cfg: LIDCConfig
) -> None:
    """Save splits to local preprocessed directory and QC."""
    prep_dir = LOCAL_PREPROCESSED_ROOT / SUBSET_DATASET_NAME
    prep_dir.mkdir(parents=True, exist_ok=True)

    splits_path = prep_dir / "splits_final.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    logger.info(f"Splits -> {splits_path}")

    qc_path = cfg.QC_DIR / f"nnunet_subset_splits_seed{SEED}.json"
    with open(qc_path, "w") as f:
        json.dump(splits, f, indent=2)
    logger.info(f"Splits (QC copy) -> {qc_path}")


# ── Disk budget guard ─────────────────────────────────────────────────────
def check_disk_budget(N: int, avg_npz_mb: float, stage: str) -> None:
    """Refuse to proceed if projected free space < safety threshold."""
    available_gb = get_available_disk_gb(Path("/"))
    per_case_mb = avg_npz_mb * (1 + UNPACK_FACTOR)
    projected_gb = N * per_case_mb / 1000

    remaining_gb = available_gb - projected_gb
    logger.info(
        f"[DISK GUARD - {stage}] Available: {available_gb:.1f} GB, "
        f"Projected: {projected_gb:.1f} GB, "
        f"Remaining: {remaining_gb:.1f} GB"
    )

    if remaining_gb < SAFETY_MARGIN_GB:
        raise RuntimeError(
            f"DISK BUDGET EXCEEDED at {stage}! "
            f"Available={available_gb:.1f} GB, Projected={projected_gb:.1f} GB, "
            f"Remaining={remaining_gb:.1f} GB < margin={SAFETY_MARGIN_GB} GB. "
            f"Reduce N or free disk space."
        )


# ── Step 6: Preprocess ────────────────────────────────────────────────────
def run_preprocessing(cfg: LIDCConfig) -> None:
    """Run nnUNetv2_plan_and_preprocess with local preprocessed dir."""
    env = os.environ.copy()
    env["nnUNet_raw"] = str(cfg.NNUNET_RAW)
    env["nnUNet_preprocessed"] = str(LOCAL_PREPROCESSED_ROOT)
    env["nnUNet_results"] = str(LOCAL_RESULTS_ROOT)

    LOCAL_PREPROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    LOCAL_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", str(SUBSET_DATASET_ID),
        "--verify_dataset_integrity",
        "-c", "3d_fullres",
        "--clean",
    ]
    logger.info(f"Running: {' '.join(cmd)}")
    logger.info(f"  nnUNet_raw = {cfg.NNUNET_RAW}")
    logger.info(f"  nnUNet_preprocessed = {LOCAL_PREPROCESSED_ROOT}")

    disk_before = get_available_disk_gb(Path("/"))
    logger.info(f"Disk before preprocessing: {disk_before:.1f} GB free")

    proc = subprocess.run(cmd, env=env, timeout=14400)  # 4h timeout
    if proc.returncode != 0:
        raise RuntimeError(f"Preprocessing failed with code {proc.returncode}")

    disk_after = get_available_disk_gb(Path("/"))
    logger.info(f"Disk after preprocessing: {disk_after:.1f} GB free (used {disk_before - disk_after:.1f} GB)")


# ── Step 7: Train ─────────────────────────────────────────────────────────
def run_training(cfg: LIDCConfig, folds: List[int]) -> None:
    """Run nnUNet training for specified folds."""
    env = os.environ.copy()
    env["nnUNet_raw"] = str(cfg.NNUNET_RAW)
    env["nnUNet_preprocessed"] = str(LOCAL_PREPROCESSED_ROOT)
    env["nnUNet_results"] = str(LOCAL_RESULTS_ROOT)

    for fold in folds:
        logger.info(f"Training fold {fold}...")
        disk_before = get_available_disk_gb(Path("/"))
        logger.info(f"Disk before training fold {fold}: {disk_before:.1f} GB free")

        if disk_before < SAFETY_MARGIN_GB:
            raise RuntimeError(
                f"Insufficient disk for training! {disk_before:.1f} GB < {SAFETY_MARGIN_GB} GB"
            )

        cmd = [
            "nnUNetv2_train",
            str(SUBSET_DATASET_ID),
            "3d_fullres",
            str(fold),
            "--npz",
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, env=env, timeout=86400)  # 24h timeout
        if proc.returncode != 0:
            raise RuntimeError(f"Training fold {fold} failed with code {proc.returncode}")

        disk_after = get_available_disk_gb(Path("/"))
        logger.info(f"Disk after training fold {fold}: {disk_after:.1f} GB free")


# ── Step 8: OOF prediction ────────────────────────────────────────────────
def run_oof_prediction(
    cfg: LIDCConfig,
    splits: List[Dict[str, List[str]]],
    folds: List[int],
    device: str = "cuda:0",
) -> List[Dict[str, Any]]:
    """Run OOF predictions for trained folds."""
    import tempfile

    env = os.environ.copy()
    env["nnUNet_raw"] = str(cfg.NNUNET_RAW)
    env["nnUNet_preprocessed"] = str(LOCAL_PREPROCESSED_ROOT)
    env["nnUNet_results"] = str(LOCAL_RESULTS_ROOT)

    oof_dir = cfg.QC_DIR / "nnunet_subset_oof_predictions"
    oof_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []
    mapping = []

    for fold in folds:
        # Check checkpoint exists
        ckpt = (
            LOCAL_RESULTS_ROOT
            / SUBSET_DATASET_NAME
            / "nnUNetTrainer__nnUNetPlans__3d_fullres"
            / f"fold_{fold}"
            / "checkpoint_final.pth"
        )
        if not ckpt.exists():
            logger.warning(f"Fold {fold} checkpoint not found, skipping")
            continue

        val_cases = splits[fold]["val"]
        logger.info(f"Predicting fold {fold}: {len(val_cases)} val cases")
        t0 = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            # Symlink validation images
            images_tr = cfg.NNUNET_RAW / SUBSET_DATASET_NAME / "imagesTr"
            for cn in val_cases:
                src = images_tr / f"{cn}_0000.nii.gz"
                dst = input_dir / f"{cn}_0000.nii.gz"
                if src.exists():
                    dst.symlink_to(src.resolve())
                else:
                    logger.warning(f"Missing image for {cn}")

            cmd = [
                "nnUNetv2_predict",
                "-i", str(input_dir),
                "-o", str(output_dir),
                "-d", str(SUBSET_DATASET_ID),
                "-c", "3d_fullres",
                "-f", str(fold),
                "-p", "nnUNetPlans",
                "-device", device,
            ]
            proc = subprocess.run(cmd, env=env, timeout=7200)
            if proc.returncode != 0:
                logger.error(f"Prediction failed for fold {fold}")
                fold_results.append({"fold": fold, "success": False})
                continue

            # Copy predictions out
            for cn in val_cases:
                pred_src = output_dir / f"{cn}.nii.gz"
                pred_dst = oof_dir / f"{cn}.nii.gz"
                if pred_src.exists():
                    shutil.copyfile(pred_src, pred_dst)

                    num = cn.replace("LIDC", "")
                    patient_id = f"LIDC-IDRI-{num}"
                    mapping.append({
                        "case_name": cn,
                        "patient_id": patient_id,
                        "fold": str(fold),
                    })

        elapsed = time.time() - t0
        fold_results.append({
            "fold": fold,
            "success": True,
            "n_cases": len(val_cases),
            "duration_s": round(elapsed, 1),
        })
        logger.info(f"Fold {fold} predictions done in {elapsed:.0f}s")

    # Save mapping
    mapping_path = cfg.QC_DIR / "nnunet_subset_oof_mapping.csv"
    with open(mapping_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_name", "patient_id", "fold"])
        writer.writeheader()
        writer.writerows(mapping)
    logger.info(f"OOF mapping -> {mapping_path}")

    return fold_results


# ── Step 9: Evaluate ──────────────────────────────────────────────────────
def evaluate_oof(cfg: LIDCConfig) -> Dict[str, Any]:
    """Compute metrics: Dice, HD95, volume stats."""
    import SimpleITK as sitk
    from scipy.ndimage import binary_erosion, distance_transform_edt

    oof_dir = cfg.QC_DIR / "nnunet_subset_oof_predictions"
    mapping_path = cfg.QC_DIR / "nnunet_subset_oof_mapping.csv"

    with open(mapping_path) as f:
        reader = csv.DictReader(f)
        mapping = list(reader)

    results = []
    for row in mapping:
        pid = row["patient_id"]
        cn = row["case_name"]
        gt_path = cfg.CASES_DIR / pid / "seg" / "nodule_mask_gt.nii.gz"
        pred_path = oof_dir / f"{cn}.nii.gz"

        if not gt_path.exists() or not pred_path.exists():
            results.append({"patient_id": pid, "error": "file_missing"})
            continue

        try:
            gt_img = sitk.ReadImage(str(gt_path))
            pred_img = sitk.ReadImage(str(pred_path))
            gt = sitk.GetArrayFromImage(gt_img).astype(bool)
            pred = sitk.GetArrayFromImage(pred_img).astype(bool)
            spacing = gt_img.GetSpacing()[::-1]  # zyx

            if pred.shape != gt.shape:
                results.append({"patient_id": pid, "error": f"shape_mismatch"})
                continue

            # Dice
            intersection = np.sum(pred & gt)
            total = np.sum(pred) + np.sum(gt)
            dice = 2 * intersection / total if total > 0 else (1.0 if np.sum(pred) == 0 else 0.0)

            # HD95
            hd95 = float("nan")
            if np.sum(pred) > 0 and np.sum(gt) > 0:
                pred_surf = pred & ~binary_erosion(pred, iterations=1)
                gt_surf = gt & ~binary_erosion(gt, iterations=1)
                if np.sum(pred_surf) == 0:
                    pred_surf = pred
                if np.sum(gt_surf) == 0:
                    gt_surf = gt
                d_from_gt = distance_transform_edt(~gt_surf, sampling=spacing)
                d_from_pred = distance_transform_edt(~pred_surf, sampling=spacing)
                hd95 = float(max(
                    np.percentile(d_from_gt[pred_surf], 95),
                    np.percentile(d_from_pred[gt_surf], 95),
                ))

            # Volume
            voxel_vol = float(np.prod(spacing))
            gt_vol_ml = float(np.sum(gt)) * voxel_vol / 1000
            pred_vol_ml = float(np.sum(pred)) * voxel_vol / 1000

            # Precision / Recall
            tp = np.sum(pred & gt)
            fp = np.sum(pred & ~gt)
            fn = np.sum(~pred & gt)
            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

            results.append({
                "patient_id": pid,
                "dice": round(float(dice), 6),
                "hd95_mm": round(hd95, 2) if not np.isnan(hd95) else None,
                "precision": round(prec, 6),
                "recall": round(rec, 6),
                "gt_volume_ml": round(gt_vol_ml, 4),
                "pred_volume_ml": round(pred_vol_ml, 4),
                "volume_diff_ml": round(pred_vol_ml - gt_vol_ml, 4),
                "error": None,
            })
        except Exception as e:
            results.append({"patient_id": pid, "error": str(e)})

    # Save CSV
    csv_path = cfg.QC_DIR / "metrics_nnunet_subset_oof_vs_gt.csv"
    fieldnames = [
        "patient_id", "dice", "hd95_mm", "precision", "recall",
        "gt_volume_ml", "pred_volume_ml", "volume_diff_ml", "error",
    ]
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
        gt_vols = [r["gt_volume_ml"] for r in valid]
        pred_vols = [r["pred_volume_ml"] for r in valid]
        vol_diffs = [r["volume_diff_ml"] for r in valid]

        summary = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": SUBSET_DATASET_NAME,
            "seed": SEED,
            "n_cases_total": len(results),
            "n_cases_valid": len(valid),
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
            "volume_ml": {
                "gt_mean": round(float(np.mean(gt_vols)), 4),
                "pred_mean": round(float(np.mean(pred_vols)), 4),
                "diff_mean": round(float(np.mean(vol_diffs)), 4),
                "diff_std": round(float(np.std(vol_diffs)), 4),
            },
        }
    else:
        summary = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "n_cases_valid": 0,
            "error": "no_valid_results",
        }

    summary_path = cfg.QC_DIR / "metrics_nnunet_subset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary -> {summary_path}")

    # Print
    if valid:
        logger.info(
            f"\n{'='*60}\n"
            f"SUBSET EVALUATION RESULTS ({SUBSET_DATASET_NAME})\n"
            f"{'='*60}\n"
            f"Cases: {len(valid)}/{len(results)} valid\n"
            f"Dice:      {summary['dice']['mean']:.4f} +/- {summary['dice']['std']:.4f} "
            f"(median {summary['dice']['median']:.4f})\n"
            f"HD95 (mm): {summary['hd95_mm']['mean']:.2f} +/- {summary['hd95_mm']['std']:.2f}\n"
            f"Volume:    GT={summary['volume_ml']['gt_mean']:.3f} ml, "
            f"Pred={summary['volume_ml']['pred_mean']:.3f} ml\n"
        )

    return summary


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train nnU-Net on LIDC subset")
    parser.add_argument(
        "--step",
        choices=["all", "select", "create_dataset", "preprocess", "train", "predict", "evaluate"],
        default="all",
        help="Which step to run (default: all)",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="0",
        help="Comma-separated folds to train/predict (default: 0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="GPU device (default: cuda:0)",
    )
    parser.add_argument(
        "--n-override",
        type=int,
        default=None,
        help="Override computed N (for testing)",
    )
    args = parser.parse_args()

    folds = [int(f) for f in args.folds.split(",")]
    cfg = get_cfg()
    steps = (
        ["select", "create_dataset", "preprocess", "train", "predict", "evaluate"]
        if args.step == "all"
        else [args.step]
    )

    logger.info(f"Steps: {steps}, Folds: {folds}")
    logger.info(f"Disk free: {get_available_disk_gb(Path('/')):.1f} GB")

    # ── Select ────────────────────────────────────
    selected_ids = None
    if "select" in steps or "create_dataset" in steps or "preprocess" in steps:
        stats = measure_per_case_footprint(cfg)
        calc = compute_subset_size(stats, n_folds_to_train=len(folds))
        N = args.n_override or calc["max_N"]
        logger.info(f"Chosen N = {N}")

        if "select" in steps:
            selected_ids = select_subset(cfg, N, stats, calc)

    # Load selected IDs if not just computed
    if selected_ids is None:
        ids_path = cfg.QC_DIR / "nnunet_subset_case_ids_seed42.txt"
        if ids_path.exists():
            selected_ids = [l.strip() for l in ids_path.read_text().splitlines() if l.strip()]
        else:
            raise FileNotFoundError(f"No selected IDs found at {ids_path}. Run --step select first.")

    # ── Create dataset ────────────────────────────
    if "create_dataset" in steps:
        create_subset_dataset(cfg, selected_ids)
        splits = generate_subset_splits(selected_ids)
        save_subset_splits(splits, cfg)

    # ── Preprocess ────────────────────────────────
    if "preprocess" in steps:
        check_disk_budget(len(selected_ids), stats["avg_npz_mb"], "pre-preprocessing")
        run_preprocessing(cfg)

    # ── Train ─────────────────────────────────────
    if "train" in steps:
        check_disk_budget(
            len(selected_ids),
            measure_per_case_footprint(cfg)["avg_npz_mb"]
            if "select" not in steps
            else stats["avg_npz_mb"],
            "pre-training",
        )
        run_training(cfg, folds)

    # ── Predict ───────────────────────────────────
    if "predict" in steps:
        splits_path = LOCAL_PREPROCESSED_ROOT / SUBSET_DATASET_NAME / "splits_final.json"
        if not splits_path.exists():
            splits_path = cfg.QC_DIR / f"nnunet_subset_splits_seed{SEED}.json"
        with open(splits_path) as f:
            splits = json.load(f)
        run_oof_prediction(cfg, splits, folds, device=args.device)

    # ── Evaluate ──────────────────────────────────
    if "evaluate" in steps:
        evaluate_oof(cfg)

    logger.info("Done!")


if __name__ == "__main__":
    main()
