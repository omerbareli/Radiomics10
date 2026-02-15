# pipeline/lidc/predict_from_preprocessed.py
"""Fast prediction using locally-cached preprocessed npz files.

Bypasses the slow mount I/O by loading preprocessed data directly,
running the model, and exporting predictions as NIfTI.

Uses nnUNet v2.1 internal functions (no nnUNetPredictor class in this version).

Usage:
    python -m pipeline.lidc.predict_from_preprocessed \
        --fold 0 --device cuda:0 --checkpoint checkpoint_best.pth
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATASET_NAME = "Dataset503_LIDC_SUBSET"
LOCAL_PREP = Path("/home/asafz/projects/radiomics10/nnUNet_preprocessed_local")
LOCAL_RES = Path("/home/asafz/projects/radiomics10/nnUNet_results_local")
QC_DIR = Path("/mnt/metlab27/D/Input/Radiomics/T_IL Radiomics10/LIDC-IDRI/Output/qc")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_best.pth")
    parser.add_argument("--epoch-label", type=str, default=None,
                        help="Epoch label for output dir naming (e.g. 226)")
    parser.add_argument("--no-mirroring", action="store_true",
                        help="Disable test-time mirroring for ~8x speedup")
    parser.add_argument("--tile-step", type=float, default=0.5,
                        help="Tile step size (0.5=default, 0.75=faster)")
    args = parser.parse_args()

    from nnunetv2.inference.export_prediction import export_prediction_from_softmax
    from nnunetv2.inference.predict_from_raw_data import load_what_we_need
    from nnunetv2.inference.sliding_window_prediction import (
        compute_gaussian,
        predict_sliding_window_return_logits,
    )
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

    # Load splits
    splits_path = LOCAL_PREP / DATASET_NAME / "splits_final.json"
    with open(splits_path) as f:
        splits = json.load(f)
    val_cases = splits[args.fold]["val"]
    logger.info(f"Fold {args.fold}: {len(val_cases)} validation cases")

    # Set up output dir
    suffix = f"_epoch{args.epoch_label}" if args.epoch_label else ""
    out_dir = QC_DIR / f"interim_predictions_fold{args.fold}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check what's already predicted (handle .nii.gz double extension)
    existing = set()
    for f in out_dir.glob("*.nii.gz"):
        # f.name is like "LIDC0001.nii.gz", stem strips last ext only
        name = f.name.replace(".nii.gz", "")
        existing.add(name)
    todo = [cn for cn in val_cases if cn not in existing]
    logger.info(f"Already predicted: {len(existing)}, remaining: {len(todo)}")
    if not todo:
        logger.info("All predictions exist, skipping")
        return

    # Load model using nnUNet v2.1 API
    model_folder = str(
        LOCAL_RES / DATASET_NAME / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    )
    device = torch.device(args.device)

    parameters, configuration_manager, inference_allowed_mirroring_axes, \
        plans_manager, dataset_json, network, trainer_name = \
        load_what_we_need(model_folder, (args.fold,), args.checkpoint)

    network = network.to(device)
    network.eval()

    # Load the single fold's weights
    network.load_state_dict(parameters[0])

    # Precompute gaussian for sliding window
    inference_gaussian = torch.from_numpy(
        compute_gaussian(configuration_manager.patch_size)
    ).half().to(device)

    label_manager = plans_manager.get_label_manager(dataset_json)
    num_seg_heads = label_manager.num_segmentation_heads

    prep_3d = LOCAL_PREP / DATASET_NAME / "nnUNetPlans_3d_fullres"

    t0 = time.time()
    for i, cn in enumerate(todo):
        case_t0 = time.time()

        npz_path = prep_3d / f"{cn}.npz"
        pkl_path = prep_3d / f"{cn}.pkl"
        if not npz_path.exists():
            logger.warning(f"Missing: {npz_path}")
            continue

        # Load preprocessed data
        data = torch.from_numpy(np.load(npz_path)["data"]).contiguous()  # (C, D, H, W)

        # Load properties (needed for resampling back to original space)
        with open(pkl_path, "rb") as f:
            props = pickle.load(f)

        # Run sliding window prediction
        mirror = None if args.no_mirroring else inference_allowed_mirroring_axes
        with torch.no_grad():
            prediction = predict_sliding_window_return_logits(
                network, data, num_seg_heads,
                configuration_manager.patch_size,
                mirror_axes=mirror,
                tile_step_size=args.tile_step,
                use_gaussian=True,
                precomputed_gaussian=inference_gaussian,
                perform_everything_on_gpu=True,
                verbose=False,
                device=device,
            )
        prediction = prediction.cpu().numpy()

        # Export: resample to original space + save as NIfTI
        out_path_truncated = str(out_dir / cn)  # without extension
        export_prediction_from_softmax(
            prediction, props, configuration_manager, plans_manager,
            dataset_json, out_path_truncated, save_probabilities=False,
        )

        elapsed = time.time() - case_t0
        logger.info(f"  [{i+1}/{len(todo)}] {cn}: {elapsed:.1f}s")

    total = time.time() - t0
    logger.info(f"Prediction complete: {len(todo)} cases in {total:.0f}s ({total/len(todo):.1f}s/case)")


if __name__ == "__main__":
    main()
