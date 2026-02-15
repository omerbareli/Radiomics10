# pipeline/lidc/prepare_heldout_input.py
"""Identify held-out OK cases (876 - 475 = ~401) and create a symlink
input directory for nnUNetv2_predict ensemble inference.

Usage:
    python -m pipeline.lidc.prepare_heldout_input
"""
from __future__ import annotations

import csv
import json
import logging
import os
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
NNUNET_RAW = OUTPUT_ROOT / "nnunet" / "nnUNet_raw"

# Dataset502 has ALL 876 OK cases
DATASET502_IMAGES = NNUNET_RAW / "Dataset502_LIDC" / "imagesTr"

# Subset case list (475 cases used for training)
SUBSET_IDS_FILE = QC_DIR / "nnunet_subset_case_ids_seed42.txt"

# Full QC report (876 OK cases)
QC_REPORT = QC_DIR / "lidc_full_qc_report.csv"

# Output directory for held-out input symlinks
HELDOUT_INPUT_DIR = OUTPUT_ROOT / "nnunet" / "heldout_input"


def patient_id_to_case_name(patient_id: str) -> str:
    """LIDC-IDRI-0001 -> LIDC0001"""
    num = patient_id.split("-")[-1]
    return f"LIDC{num}"


def main():
    # 1. Load all 876 OK patient IDs from QC report
    ok_patient_ids = set()
    with open(QC_REPORT) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["qc_flag"] == "OK":
                ok_patient_ids.add(row["patient_id"])
    logger.info(f"Total OK cases from QC report: {len(ok_patient_ids)}")

    # 2. Load 475 subset patient IDs
    subset_patient_ids = set()
    with open(SUBSET_IDS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                subset_patient_ids.add(line)
    logger.info(f"Subset cases (training): {len(subset_patient_ids)}")

    # 3. Compute held-out set
    heldout_patient_ids = sorted(ok_patient_ids - subset_patient_ids)
    logger.info(f"Held-out cases: {len(heldout_patient_ids)}")

    # 4. Create symlink input directory
    HELDOUT_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clean existing symlinks
    existing = list(HELDOUT_INPUT_DIR.glob("*.nii.gz"))
    if existing:
        logger.info(f"Clearing {len(existing)} existing files in {HELDOUT_INPUT_DIR}")
        for f in existing:
            f.unlink()

    linked = 0
    missing = []
    for pid in heldout_patient_ids:
        cn = patient_id_to_case_name(pid)
        src = DATASET502_IMAGES / f"{cn}_0000.nii.gz"
        dst = HELDOUT_INPUT_DIR / f"{cn}_0000.nii.gz"

        if not src.exists():
            missing.append(pid)
            continue

        # Symlink on same filesystem (mount -> mount)
        try:
            dst.symlink_to(src)
            linked += 1
        except OSError:
            # Fallback: hardlink or copy
            try:
                os.link(str(src), str(dst))
                linked += 1
            except OSError:
                missing.append(pid)

    logger.info(f"Created {linked} symlinks in {HELDOUT_INPUT_DIR}")
    if missing:
        logger.warning(f"Missing images for {len(missing)} cases: {missing[:10]}...")

    # 5. Save held-out case list
    heldout_ids_file = QC_DIR / "heldout_case_ids.txt"
    with open(heldout_ids_file, "w") as f:
        for pid in heldout_patient_ids:
            f.write(pid + "\n")
    logger.info(f"Held-out case list -> {heldout_ids_file}")

    # 6. Save metadata
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_ok_cases": len(ok_patient_ids),
        "subset_cases": len(subset_patient_ids),
        "heldout_cases": len(heldout_patient_ids),
        "images_linked": linked,
        "images_missing": len(missing),
        "input_dir": str(HELDOUT_INPUT_DIR),
        "source_dataset": "Dataset502_LIDC",
    }
    metadata_file = QC_DIR / "heldout_selection.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata -> {metadata_file}")

    logger.info(
        f"\nReady for ensemble inference:\n"
        f"  nnUNetv2_predict -i {HELDOUT_INPUT_DIR} -o <output_dir> "
        f"-d 503 -c 3d_fullres -f 0 1 2 3 --disable_tta"
    )


if __name__ == "__main__":
    main()
