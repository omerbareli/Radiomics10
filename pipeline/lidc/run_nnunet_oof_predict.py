# pipeline/lidc/run_nnunet_oof_predict.py
"""Generate out-of-fold (OOF) predictions using trained nnU-Net models.

For each fold, runs prediction on that fold's validation set using the model
trained on the other folds. This ensures leakage-free evaluation.

Usage:
    python -m pipeline.lidc.run_nnunet_oof_predict --seed 42
    
Prerequisites:
    - All 5 folds trained successfully
    - Environment variables set: nnUNet_raw, nnUNet_preprocessed, nnUNet_results
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.lidc.config import LIDCConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


DATASET_ID = 502
DATASET_NAME = "Dataset502_LIDC"


def get_nnunet_paths() -> Dict[str, Path]:
    """Get nnU-Net paths from environment variables."""
    required = ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]
    paths = {}
    for var in required:
        val = os.environ.get(var)
        if not val:
            raise RuntimeError(f"Environment variable {var} not set")
        paths[var] = Path(val)
    return paths


def load_splits(cfg: LIDCConfig, seed: int) -> List[Dict[str, List[str]]]:
    """Load the splits from QC directory."""
    splits_path = cfg.QC_DIR / f"nnunet_splits_seed{seed}.json"
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_path}")
    with open(splits_path) as f:
        return json.load(f)


def check_fold_trained(results_path: Path, fold: int) -> bool:
    """Check if a fold completed training."""
    fold_dir = (
        results_path
        / DATASET_NAME
        / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        / f"fold_{fold}"
    )
    checkpoint = fold_dir / "checkpoint_final.pth"
    return checkpoint.exists()


def run_prediction_for_fold(
    fold: int,
    val_cases: List[str],
    nnunet_paths: Dict[str, Path],
    output_dir: Path,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """Run prediction for a single fold's validation cases.
    
    Returns:
        Dict with fold, num_cases, success, duration_seconds, errors
    """
    import time
    start_time = time.time()
    
    result = {
        "fold": fold,
        "num_cases": len(val_cases),
        "success": False,
        "duration_seconds": 0,
        "errors": [],
    }
    
    # Check model exists
    results_path = nnunet_paths["nnUNet_results"]
    if not check_fold_trained(results_path, fold):
        result["errors"].append(f"Fold {fold} not trained")
        return result
    
    # Create temp input/output directories
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_dir = tmpdir / "input"
        pred_dir = tmpdir / "output"
        input_dir.mkdir()
        pred_dir.mkdir()
        
        # Symlink validation images to temp input
        images_tr = nnunet_paths["nnUNet_raw"] / DATASET_NAME / "imagesTr"
        for case_name in val_cases:
            src = images_tr / f"{case_name}_0000.nii.gz"
            dst = input_dir / f"{case_name}_0000.nii.gz"
            if src.exists():
                dst.symlink_to(src)
            else:
                result["errors"].append(f"Missing image: {case_name}")
        
        if result["errors"]:
            return result
        
        # Run nnUNetv2_predict
        model_folder = (
            results_path
            / DATASET_NAME
            / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        )
        
        cmd = [
            "nnUNetv2_predict",
            "-i", str(input_dir),
            "-o", str(pred_dir),
            "-d", str(DATASET_ID),
            "-c", "3d_fullres",
            "-f", str(fold),
            "-p", "nnUNetPlans",
            "--save_probabilities",
            "-device", device,
        ]
        
        logger.info(f"Running fold {fold} prediction: {len(val_cases)} cases")
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout per fold
            )
            if proc.returncode != 0:
                result["errors"].append(f"Prediction failed: {proc.stderr[-500:]}")
                return result
        except subprocess.TimeoutExpired:
            result["errors"].append("Prediction timed out (>2h)")
            return result
        except Exception as e:
            result["errors"].append(f"Prediction error: {e}")
            return result
        
        # Copy predictions to final output
        for case_name in val_cases:
            pred_src = pred_dir / f"{case_name}.nii.gz"
            pred_dst = output_dir / f"{case_name}.nii.gz"
            if pred_src.exists():
                shutil.copyfile(pred_src, pred_dst)
            else:
                result["errors"].append(f"No prediction for {case_name}")
    
    result["success"] = len(result["errors"]) == 0
    result["duration_seconds"] = time.time() - start_time
    return result


def copy_predictions_to_cases(
    cfg: LIDCConfig,
    oof_output_dir: Path,
    mapping: List[Dict[str, str]],
) -> int:
    """Copy OOF predictions to individual case directories.
    
    Returns:
        Number of predictions successfully copied
    """
    copied = 0
    for row in mapping:
        case_name = row["case_name"]
        patient_id = row["patient_id"]
        
        pred_src = oof_output_dir / f"{case_name}.nii.gz"
        if not pred_src.exists():
            logger.warning(f"[{patient_id}] Prediction not found: {pred_src}")
            continue
        
        case_dir = cfg.CASES_DIR / patient_id / "seg"
        case_dir.mkdir(parents=True, exist_ok=True)
        pred_dst = case_dir / "nodule_mask_nnunet_oof.nii.gz"
        
        shutil.copyfile(pred_src, pred_dst)
        copied += 1
    
    return copied


def main():
    parser = argparse.ArgumentParser(
        description="Generate OOF predictions using trained nnU-Net models"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for splits (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference (default: cuda:0)",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="all",
        help="Folds to run, comma-separated or 'all' (default: all)",
    )
    args = parser.parse_args()
    
    cfg = LIDCConfig()
    nnunet_paths = get_nnunet_paths()
    
    # Load splits
    splits = load_splits(cfg, args.seed)
    n_folds = len(splits)
    logger.info(f"Loaded {n_folds} folds from seed {args.seed}")
    
    # Determine which folds to run
    if args.folds == "all":
        folds_to_run = list(range(n_folds))
    else:
        folds_to_run = [int(f) for f in args.folds.split(",")]
    
    # Check fold training status
    for fold in folds_to_run:
        if not check_fold_trained(nnunet_paths["nnUNet_results"], fold):
            logger.warning(f"Fold {fold} not trained - skipping")
            folds_to_run.remove(fold)
    
    if not folds_to_run:
        logger.error("No trained folds found!")
        return
    
    logger.info(f"Running predictions for folds: {folds_to_run}")
    
    # Output directory for OOF predictions
    oof_output_dir = cfg.OUTPUT_ROOT / "nnunet_oof_predictions"
    oof_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mapping from case_name to patient_id and fold
    mapping: List[Dict[str, str]] = []
    fold_results: List[Dict[str, Any]] = []
    
    # Run predictions for each fold
    for fold in folds_to_run:
        val_cases = splits[fold]["val"]
        
        # Run prediction
        result = run_prediction_for_fold(
            fold=fold,
            val_cases=val_cases,
            nnunet_paths=nnunet_paths,
            output_dir=oof_output_dir,
            device=args.device,
        )
        fold_results.append(result)
        
        if result["success"]:
            logger.info(
                f"✓ Fold {fold}: {len(val_cases)} predictions "
                f"({result['duration_seconds']:.1f}s)"
            )
            # Add to mapping
            for case_name in val_cases:
                # Convert case_name (LIDC0001) back to patient_id (LIDC-IDRI-0001)
                num = case_name.replace("LIDC", "")
                patient_id = f"LIDC-IDRI-{num}"
                mapping.append({
                    "case_name": case_name,
                    "patient_id": patient_id,
                    "fold": str(fold),
                    "seed": str(args.seed),
                })
        else:
            logger.error(f"✗ Fold {fold} failed: {result['errors']}")
    
    # Save mapping
    mapping_path = cfg.QC_DIR / "nnunet_oof_mapping.csv"
    with open(mapping_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_name", "patient_id", "fold", "seed"])
        writer.writeheader()
        writer.writerows(mapping)
    logger.info(f"✓ OOF mapping: {mapping_path}")
    
    # Copy predictions to case directories
    copied = copy_predictions_to_cases(cfg, oof_output_dir, mapping)
    logger.info(f"✓ Copied {copied}/{len(mapping)} predictions to case directories")
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
        "n_folds": n_folds,
        "folds_run": folds_to_run,
        "total_predictions": len(mapping),
        "predictions_copied": copied,
        "fold_results": fold_results,
    }
    summary_path = cfg.QC_DIR / f"nnunet_oof_summary_seed{args.seed}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"✓ Summary: {summary_path}")
    
    # Final status
    successful_folds = sum(1 for r in fold_results if r["success"])
    logger.info(
        f"\n{'='*50}\n"
        f"OOF PREDICTIONS COMPLETE\n"
        f"{'='*50}\n"
        f"Folds: {successful_folds}/{len(folds_to_run)} successful\n"
        f"Predictions: {len(mapping)}\n"
        f"Copied to cases: {copied}\n"
    )


if __name__ == "__main__":
    main()
