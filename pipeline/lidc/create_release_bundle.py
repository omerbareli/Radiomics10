# pipeline/lidc/create_release_bundle.py
"""Create a clean release bundle on the mount with all model artifacts,
evaluation results, and reproduction instructions.

Run AFTER all folds are trained and evaluations are complete.

Usage:
    python -m pipeline.lidc.create_release_bundle
    python -m pipeline.lidc.create_release_bundle --folds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

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
NNUNET_OUTPUT = OUTPUT_ROOT / "nnunet"

LOCAL_RES = Path("/home/asafz/projects/radiomics10/nnUNet_results_local")
LOCAL_PREP = Path("/home/asafz/projects/radiomics10/nnUNet_preprocessed_local")
TRAINER_DIR = (
    LOCAL_RES / "Dataset503_LIDC_SUBSET" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
)

RELEASE_DIR = OUTPUT_ROOT / "release_nnunet_503"


def copy_if_exists(src: Path, dst: Path, label: str = "") -> bool:
    if not src.exists():
        logger.warning(f"  SKIP (missing): {label or src.name}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    logger.info(f"  OK: {label or src.name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Create release bundle on mount")
    parser.add_argument(
        "--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
        help="Folds to include (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be copied without copying",
    )
    args = parser.parse_args()

    folds = sorted(args.folds)
    logger.info(f"Creating release bundle at {RELEASE_DIR}")
    logger.info(f"Folds: {folds}")

    if args.dry_run:
        logger.info("DRY RUN - no files will be copied")

    RELEASE_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Model artifacts ─────────────────────────────────────────────
    logger.info("\n--- Model Artifacts ---")
    model_dir = RELEASE_DIR / "model"

    # Plans and dataset config
    for name in ["plans.json", "dataset.json", "dataset_fingerprint.json"]:
        src = TRAINER_DIR.parent / name
        if not args.dry_run:
            copy_if_exists(src, model_dir / name, name)
        else:
            logger.info(f"  Would copy: {name} ({'exists' if src.exists() else 'MISSING'})")

    # Per-fold checkpoints
    for fold in folds:
        fold_dir = TRAINER_DIR / f"fold_{fold}"
        for ckpt in ["checkpoint_final.pth", "checkpoint_best.pth"]:
            src = fold_dir / ckpt
            dst = model_dir / f"fold_{fold}" / ckpt
            if not args.dry_run:
                copy_if_exists(src, dst, f"fold_{fold}/{ckpt}")
            else:
                logger.info(
                    f"  Would copy: fold_{fold}/{ckpt} "
                    f"({'exists' if src.exists() else 'MISSING'})"
                )

        # Training log and progress plot
        for name in ["progress.png", "debug.json"]:
            src = fold_dir / name
            if not args.dry_run:
                copy_if_exists(
                    src, model_dir / f"fold_{fold}" / name,
                    f"fold_{fold}/{name}",
                )

        # Copy training logs
        for logfile in fold_dir.glob("training_log_*.txt"):
            if not args.dry_run:
                copy_if_exists(
                    logfile,
                    model_dir / f"fold_{fold}" / logfile.name,
                    f"fold_{fold}/{logfile.name}",
                )

    # ── 2. Evaluation results ──────────────────────────────────────────
    logger.info("\n--- Evaluation Results ---")
    eval_dir = RELEASE_DIR / "evaluation"

    # OOF metrics
    n_folds = len(folds)
    for pattern in [
        f"metrics_nnunet_oof_{n_folds}fold_final.csv",
        f"metrics_nnunet_oof_{n_folds}fold_final_summary.json",
    ]:
        src = QC_DIR / pattern
        if not args.dry_run:
            copy_if_exists(src, eval_dir / "oof" / pattern, pattern)

    # Also check 4-fold if we're building 5-fold
    if n_folds == 5:
        for pattern in [
            "metrics_nnunet_oof_4fold_final.csv",
            "metrics_nnunet_oof_4fold_final_summary.json",
        ]:
            src = QC_DIR / pattern
            if not args.dry_run:
                copy_if_exists(src, eval_dir / "oof" / pattern, pattern)

    # Held-out metrics
    for pattern in [
        "metrics_nnunet_heldout_4fold.csv",
        "metrics_nnunet_heldout_4fold_summary.json",
    ]:
        src = QC_DIR / pattern
        if not args.dry_run:
            copy_if_exists(src, eval_dir / "heldout" / pattern, pattern)

    # Crossval results summary
    cv_dir = TRAINER_DIR / f"crossval_results_folds_{'_'.join(str(f) for f in folds)}"
    if not cv_dir.exists():
        cv_dir = TRAINER_DIR / "crossval_results_folds_0_1_2_3"
    src = cv_dir / "summary.json"
    if not args.dry_run:
        copy_if_exists(src, eval_dir / "crossval_summary.json", "crossval_summary.json")

    # ── 3. Overlays ────────────────────────────────────────────────────
    logger.info("\n--- Overlay PNGs ---")
    overlay_release = RELEASE_DIR / "overlays"

    for dirname in [
        f"overlays_oof_{n_folds}fold_final",
        "overlays_oof_4fold_final",
        "overlays_metrics_nnunet_heldout_4fold",
    ]:
        src = QC_DIR / dirname
        if src.exists():
            if not args.dry_run:
                copy_if_exists(src, overlay_release / dirname, dirname)
            else:
                n_files = len(list(src.glob("*.png")))
                logger.info(f"  Would copy: {dirname}/ ({n_files} PNGs)")

    # ── 4. Config / reproducibility ────────────────────────────────────
    logger.info("\n--- Config & Reproducibility ---")
    config_dir = RELEASE_DIR / "config"

    # Splits
    src = LOCAL_PREP / "Dataset503_LIDC_SUBSET" / "splits_final.json"
    if not args.dry_run:
        copy_if_exists(src, config_dir / "splits_final.json", "splits_final.json")

    # Subset selection metadata
    for name in [
        "nnunet_subset_selection.json",
        "nnunet_subset_case_ids_seed42.txt",
        "heldout_case_ids.txt",
        "heldout_selection.json",
    ]:
        src = QC_DIR / name
        if not args.dry_run:
            copy_if_exists(src, config_dir / name, name)

    # Pascal GPU patches documentation
    patches_src = Path(
        "/home/asafz/.claude/projects/"
        "-home-asafz-projects-radiomics10-RADIOMICS10/memory/nnunet_patches.md"
    )
    if not args.dry_run:
        copy_if_exists(src=patches_src, dst=config_dir / "pascal_gpu_patches.md",
                       label="pascal_gpu_patches.md")

    # ── 5. Generate README ─────────────────────────────────────────────
    logger.info("\n--- README ---")
    readme_content = f"""# nnU-Net Release Bundle: Dataset503_LIDC_SUBSET

## Model Card

- **Task**: Lung nodule segmentation (binary: background + nodule)
- **Dataset**: LIDC-IDRI, 475-case subset (seed=42) from 876 OK cases
- **Architecture**: nnU-Net v2.1, 3d_fullres, PlainConvUNet
- **Patch size**: [96, 160, 160]
- **Spacing**: [1.25, 0.703125, 0.703125] mm
- **Training**: 1000 epochs per fold, batch_size=2
- **Folds**: {folds}
- **GPU**: NVIDIA TITAN Xp (12 GB) -- Pascal architecture
- **Mixed precision**: DISABLED (cuDNN fp16 incompatible with Pascal)
- **Created**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

## Directory Structure

```
model/                      # Trained model weights
    plans.json              # nnU-Net plans
    dataset.json            # Dataset configuration
    fold_{{0..{max(folds)}}}/
        checkpoint_final.pth   # Final weights (epoch 1000)
        checkpoint_best.pth    # Best validation checkpoint
        progress.png           # Training curves
        training_log_*.txt     # Training logs

evaluation/                 # Metrics and summaries
    oof/                    # Out-of-fold (leakage-free) evaluation
    heldout/                # Held-out cases (not used in training)
    crossval_summary.json   # nnU-Net built-in crossval

overlays/                   # QC visualization PNGs

config/                     # Reproducibility artifacts
    splits_final.json       # 5-fold split definitions
    nnunet_subset_selection.json  # Subset selection metadata
    pascal_gpu_patches.md   # GPU compatibility patches
```

## Reproduction

### Environment
```bash
# nnU-Net v2.1, PyTorch 2.1.2+cu118
pip install nnunetv2==2.1
# Apply Pascal GPU patches (see config/pascal_gpu_patches.md)
```

### Environment Variables
```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

### Training
```bash
# Per fold (one GPU each):
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 503 3d_fullres 0 --npz
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 503 3d_fullres 1 --npz
# ... etc for folds 2-4
```

### Inference (single fold)
```bash
nnUNetv2_predict -i INPUT -o OUTPUT -d 503 -c 3d_fullres -f 0
```

### Inference (ensemble)
```bash
nnUNetv2_predict -i INPUT -o OUTPUT -d 503 -c 3d_fullres -f 0 1 2 3 4
```
"""
    if not args.dry_run:
        readme_path = RELEASE_DIR / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)
        logger.info(f"  OK: README.md")

    # ── Summary ────────────────────────────────────────────────────────
    if not args.dry_run:
        # Count what was copied
        total_files = sum(1 for _ in RELEASE_DIR.rglob("*") if _.is_file())
        total_size = sum(f.stat().st_size for f in RELEASE_DIR.rglob("*") if f.is_file())
        logger.info(
            f"\nRelease bundle created at:\n"
            f"  {RELEASE_DIR}\n"
            f"  Files: {total_files}\n"
            f"  Size: {total_size / 1e9:.2f} GB\n"
        )
    else:
        logger.info("\nDry run complete. No files were copied.")


if __name__ == "__main__":
    main()
