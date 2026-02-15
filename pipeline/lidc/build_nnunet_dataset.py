# pipeline/lidc/build_nnunet_dataset.py
"""Build nnU-Net dataset from LIDC-IDRI OK cases only.

Creates:
- Dataset502_LIDC with symlinks to CT images and GT masks
- Strict nnU-Net v2 dataset.json
- Reproducible 5-fold CV splits with verification

Usage:
    python -m pipeline.lidc.build_nnunet_dataset --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from pipeline.lidc.config import LIDCConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


DATASET_ID = 502
DATASET_NAME = "Dataset502_LIDC"


def get_ok_case_ids(cfg: LIDCConfig) -> List[str]:
    """Load OK case IDs from full QC report."""
    qc_path = cfg.QC_DIR / "lidc_full_qc_report.csv"
    if not qc_path.exists():
        raise FileNotFoundError(f"QC report not found: {qc_path}")
    
    df = pd.read_csv(qc_path)
    ok_df = df[df["qc_flag"] == "OK"]
    
    # Verify CT and mask exist
    valid_ids = []
    for _, row in ok_df.iterrows():
        ct_path = Path(row.get("ct_path", ""))
        mask_path = Path(row.get("mask_path", ""))
        if ct_path.exists() and mask_path.exists():
            valid_ids.append(row["patient_id"])
        else:
            logger.warning(f"[{row['patient_id']}] Missing files, skipping")
    
    return sorted(valid_ids)


def create_symlink(src: Path, dst: Path) -> bool:
    """Create symlink, return True if successful."""
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
        return True
    except OSError:
        return False


def copy_file(src: Path, dst: Path) -> bool:
    """Copy file, return True if successful."""
    import shutil
    try:
        if dst.exists():
            dst.unlink()
        # Use copyfile (not copy2) to avoid metadata permission errors on network mounts
        shutil.copyfile(src, dst)
        return True
    except Exception as e:
        logger.warning(f"Copy failed: {e}")
        return False
        return False


def link_or_copy(src: Path, dst: Path, use_copy: bool = False) -> bool:
    """Try symlink first, fall back to copy if needed."""
    if not use_copy:
        if create_symlink(src, dst):
            return True
    # Fall back to copy
    return copy_file(src, dst)


def build_dataset_structure(
    cfg: LIDCConfig,
    ok_ids: List[str],
    force_copy: bool = False,
) -> Tuple[Path, int]:
    """Create nnU-Net dataset directory with symlinks or copies.
    
    Args:
        cfg: Config
        ok_ids: List of OK patient IDs
        force_copy: If True, always copy (skip symlink attempt)
        
    Returns:
        Tuple of (dataset_path, num_cases_linked)
    """
    # Setup paths
    nnunet_raw = cfg.OUTPUT_ROOT / "nnunet" / "nnUNet_raw"
    dataset_path = nnunet_raw / DATASET_NAME
    images_tr = dataset_path / "imagesTr"
    labels_tr = dataset_path / "labelsTr"
    
    # Create directories
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Building dataset at: {dataset_path}")
    
    # Test if symlinks work on this filesystem
    test_src = cfg.CASES_DIR / ok_ids[0] / "nifti" / "ct.nii.gz"
    test_dst = images_tr / ".symlink_test"
    use_copy = force_copy or not create_symlink(test_src, test_dst)
    if test_dst.exists() or test_dst.is_symlink():
        test_dst.unlink()
    
    if use_copy:
        logger.info("Symlinks not supported, using file copies (this may take a while)...")
    else:
        logger.info("Using symlinks...")
    
    from tqdm import tqdm
    
    linked = 0
    for patient_id in tqdm(ok_ids, desc="Linking" if not use_copy else "Copying"):
        # Source paths
        case_dir = cfg.CASES_DIR / patient_id
        ct_src = case_dir / "nifti" / "ct.nii.gz"
        mask_src = case_dir / "seg" / "nodule_mask_gt.nii.gz"
        
        if not ct_src.exists() or not mask_src.exists():
            logger.warning(f"[{patient_id}] Missing CT or mask, skipping")
            continue
        
        # nnU-Net naming: CASE_0000.nii.gz for images, CASE.nii.gz for labels
        case_name = patient_id.replace("LIDC-IDRI-", "LIDC")  # Shorter names
        img_dst = images_tr / f"{case_name}_0000.nii.gz"
        lbl_dst = labels_tr / f"{case_name}.nii.gz"
        
        # Link or copy
        img_ok = link_or_copy(ct_src, img_dst, use_copy=use_copy)
        lbl_ok = link_or_copy(mask_src, lbl_dst, use_copy=use_copy)
        
        if img_ok and lbl_ok:
            linked += 1
    
    logger.info(f"✓ {'Copied' if use_copy else 'Linked'} {linked}/{len(ok_ids)} cases")
    return dataset_path, linked


def generate_dataset_json(dataset_path: Path, num_training: int) -> None:
    """Generate strict nnU-Net v2 dataset.json."""
    dataset_json = {
        "dataset_name": DATASET_NAME,
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "nodule": 1},
        "numTraining": num_training,
        "file_ending": ".nii.gz",
    }
    
    json_path = dataset_path / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=2)
    
    logger.info(f"✓ dataset.json written: {json_path}")


def generate_splits(
    ok_ids: List[str],
    seed: int = 42,
    n_folds: int = 5,
) -> List[Dict[str, List[str]]]:
    """Generate reproducible k-fold CV splits.
    
    Returns:
        List of {"train": [...], "val": [...]} for each fold
    """
    random.seed(seed)
    ids = sorted(ok_ids)
    random.shuffle(ids)
    
    # Split into n_folds approximately equal parts
    fold_size = len(ids) // n_folds
    remainder = len(ids) % n_folds
    
    folds = []
    start = 0
    for i in range(n_folds):
        # Distribute remainder across first folds
        size = fold_size + (1 if i < remainder else 0)
        folds.append(ids[start:start + size])
        start += size
    
    # Build train/val splits
    splits = []
    for i in range(n_folds):
        val_ids = folds[i]
        train_ids = []
        for j in range(n_folds):
            if j != i:
                train_ids.extend(folds[j])
        
        # Convert to nnU-Net case names
        val_names = [pid.replace("LIDC-IDRI-", "LIDC") for pid in val_ids]
        train_names = [pid.replace("LIDC-IDRI-", "LIDC") for pid in train_ids]
        
        splits.append({
            "train": sorted(train_names),
            "val": sorted(val_names),
        })
    
    return splits


def verify_splits(
    splits: List[Dict[str, List[str]]],
    ok_ids: List[str],
) -> Dict[str, Any]:
    """Verify splits are correct and leakage-free.
    
    Returns:
        Verification report dict
    """
    n_folds = len(splits)
    all_ok_names = set(pid.replace("LIDC-IDRI-", "LIDC") for pid in ok_ids)
    
    report = {
        "total_cases": len(ok_ids),
        "n_folds": n_folds,
        "cases_per_fold_train": [],
        "cases_per_fold_val": [],
        "each_case_in_exactly_one_val_fold": True,
        "no_train_val_overlap_per_fold": True,
        "leakage_count": 0,
        "union_of_val_equals_all": True,
        "verification_passed": True,
    }
    
    val_counts = defaultdict(int)
    all_val_ids = set()
    
    for i, split in enumerate(splits):
        train_set = set(split["train"])
        val_set = set(split["val"])
        
        report["cases_per_fold_train"].append(len(train_set))
        report["cases_per_fold_val"].append(len(val_set))
        
        # Check no train/val overlap
        overlap = train_set & val_set
        if overlap:
            report["no_train_val_overlap_per_fold"] = False
            report["leakage_count"] += len(overlap)
            logger.error(f"Fold {i}: {len(overlap)} cases in both train and val!")
        
        # Track val appearances
        for case_id in val_set:
            val_counts[case_id] += 1
            all_val_ids.add(case_id)
    
    # Check each case in exactly one val fold
    for case_id, count in val_counts.items():
        if count != 1:
            report["each_case_in_exactly_one_val_fold"] = False
            logger.error(f"Case {case_id} appears in {count} val folds!")
    
    # Check union of val sets = all OK cases
    if all_val_ids != all_ok_names:
        report["union_of_val_equals_all"] = False
        missing = all_ok_names - all_val_ids
        extra = all_val_ids - all_ok_names
        if missing:
            logger.error(f"{len(missing)} OK cases not in any val fold")
        if extra:
            logger.error(f"{len(extra)} val cases not in OK list")
    
    # Overall pass
    report["verification_passed"] = (
        report["each_case_in_exactly_one_val_fold"] and
        report["no_train_val_overlap_per_fold"] and
        report["union_of_val_equals_all"] and
        report["leakage_count"] == 0
    )
    
    return report


def save_splits(
    splits: List[Dict[str, List[str]]],
    cfg: LIDCConfig,
    seed: int,
) -> None:
    """Save splits to nnU-Net location and QC directory."""
    # nnU-Net preprocessed location
    nnunet_preprocessed = cfg.OUTPUT_ROOT / "nnunet" / "nnUNet_preprocessed"
    nnunet_splits_dir = nnunet_preprocessed / DATASET_NAME
    nnunet_splits_dir.mkdir(parents=True, exist_ok=True)
    
    # nnU-Net expects splits_final.json
    nnunet_splits_path = nnunet_splits_dir / "splits_final.json"
    with open(nnunet_splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    logger.info(f"✓ nnU-Net splits: {nnunet_splits_path}")
    
    # Authoritative copy in QC dir
    qc_splits_path = cfg.QC_DIR / f"nnunet_splits_seed{seed}.json"
    with open(qc_splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    logger.info(f"✓ QC splits copy: {qc_splits_path}")


def select_pilot_cases(ok_ids: List[str], seed: int = 42, n: int = 50) -> List[str]:
    """Select random pilot cases reproducibly."""
    random.seed(seed)
    ids = sorted(ok_ids)
    return random.sample(ids, min(n, len(ids)))


def main():
    parser = argparse.ArgumentParser(
        description="Build nnU-Net dataset from LIDC-IDRI OK cases"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits and pilot selection (default: 42)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--pilot-size",
        type=int,
        default=50,
        help="Number of pilot cases (default: 50)",
    )
    args = parser.parse_args()
    
    cfg = LIDCConfig()
    
    # Step 1: Get OK case IDs
    logger.info("Loading OK case IDs...")
    ok_ids = get_ok_case_ids(cfg)
    logger.info(f"Found {len(ok_ids)} OK cases with valid files")
    
    # Save OK case IDs
    ok_ids_path = cfg.QC_DIR / "ok_case_ids_used_for_training.txt"
    ok_ids_path.write_text("\n".join(ok_ids) + "\n")
    logger.info(f"✓ OK case IDs: {ok_ids_path}")
    
    # Step 2: Build dataset structure
    dataset_path, num_linked = build_dataset_structure(cfg, ok_ids)
    
    # Step 3: Generate dataset.json
    generate_dataset_json(dataset_path, num_linked)
    
    # Step 4: Generate and verify splits
    logger.info(f"Generating {args.n_folds}-fold splits with seed={args.seed}...")
    splits = generate_splits(ok_ids, seed=args.seed, n_folds=args.n_folds)
    
    # Verify splits
    verification = verify_splits(splits, ok_ids)
    
    if verification["verification_passed"]:
        logger.info("✓ Split verification PASSED")
    else:
        logger.error("✗ Split verification FAILED")
    
    # Save verification report
    verification_path = cfg.QC_DIR / f"nnunet_split_verification_seed{args.seed}.json"
    with open(verification_path, "w") as f:
        json.dump(verification, f, indent=2)
    logger.info(f"✓ Verification report: {verification_path}")
    
    # Save splits
    save_splits(splits, cfg, args.seed)
    
    # Step 5: Select and save pilot cases
    pilot_ids = select_pilot_cases(ok_ids, seed=args.seed, n=args.pilot_size)
    pilot_path = cfg.QC_DIR / f"nnunet_pilot_case_ids_seed{args.seed}.txt"
    pilot_path.write_text("\n".join(pilot_ids) + "\n")
    logger.info(f"✓ Pilot cases ({len(pilot_ids)}): {pilot_path}")
    
    # Summary
    logger.info(
        f"\n{'='*50}\n"
        f"DATASET BUILD COMPLETE\n"
        f"{'='*50}\n"
        f"Dataset: {DATASET_NAME}\n"
        f"Location: {dataset_path}\n"
        f"Cases linked: {num_linked}\n"
        f"Folds: {args.n_folds}\n"
        f"Seed: {args.seed}\n"
        f"Pilot: {len(pilot_ids)} cases\n"
        f"Verification: {'PASSED' if verification['verification_passed'] else 'FAILED'}\n"
    )
    
    # Print environment setup
    nnunet_raw = cfg.OUTPUT_ROOT / "nnunet" / "nnUNet_raw"
    nnunet_preprocessed = cfg.OUTPUT_ROOT / "nnunet" / "nnUNet_preprocessed"
    nnunet_results = cfg.OUTPUT_ROOT / "nnunet" / "nnUNet_results"
    
    logger.info(
        f"\nTo use this dataset, set environment variables:\n"
        f"  export nnUNet_raw={nnunet_raw}\n"
        f"  export nnUNet_preprocessed={nnunet_preprocessed}\n"
        f"  export nnUNet_results={nnunet_results}\n"
        f"\nThen run:\n"
        f"  nnUNetv2_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity\n"
        f"  nnUNetv2_train {DATASET_ID} 3d_fullres 0 --npz\n"
    )


if __name__ == "__main__":
    main()
