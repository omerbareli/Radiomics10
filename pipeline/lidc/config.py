# pipeline/lidc/config.py
"""Configuration for LIDC-IDRI pipeline.

All paths point to the METLAB27 mount. Output goes to LIDC-IDRI/Output/.
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


def _env_path(name: str, default: str) -> Path:
    """Get path from environment variable or default."""
    return Path(os.environ.get(name, default)).expanduser()


# Link strategy type
LinkStrategy = Literal["symlink", "hardlink", "copy"]


def safe_link(src: Path, dst: Path) -> LinkStrategy:
    """Create link with fallback: symlink → hardlink → copy.
    
    This handles NTFS mount constraints where symlinks may fail.
    
    Args:
        src: Source file (must exist)
        dst: Destination path
        
    Returns:
        Strategy used: "symlink", "hardlink", or "copy"
    """
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing destination if present
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    
    # 1. Try symlink (preferred - no extra disk space)
    try:
        dst.symlink_to(src)
        return "symlink"
    except OSError as e:
        logger.debug(f"Symlink failed for {dst.name}: {e}")
    
    # 2. Try hardlink (same filesystem only, no extra space)
    try:
        dst.hardlink_to(src)
        logger.warning(f"Using hardlink (symlink failed): {dst.name}")
        return "hardlink"
    except OSError as e:
        logger.debug(f"Hardlink failed for {dst.name}: {e}")
    
    # 3. Fallback to copy (LOUD warning - uses disk space)
    logger.warning(
        f"⚠️ COPYING FILE (symlink+hardlink failed): {src.name} → {dst}\n"
        f"   This uses extra disk space. Check mount permissions."
    )
    shutil.copy2(src, dst)
    return "copy"


@dataclass(frozen=True)
class LIDCConfig:
    """Configuration for LIDC-IDRI dataset."""
    
    # Project root (same as main radiomics project)
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
    
    # LIDC-IDRI root on METLAB27 mount
    LIDC_ROOT: Path = _env_path(
        "LIDC_ROOT",
        "/mnt/metlab27/D/Input/Radiomics/T_IL Radiomics10/LIDC-IDRI"
    )
    
    # Data locations
    DICOM_ROOT: Path = LIDC_ROOT / "manifest-1600709154662" / "LIDC-IDRI"
    METADATA_CSV: Path = LIDC_ROOT / "manifest-1600709154662" / "metadata.csv"
    ANNOTATIONS_ZIP: Path = LIDC_ROOT / "Annotations" / "LIDC-XML-only.zip"
    
    # Metadata files
    NODULE_COUNTS_XLSX: Path = LIDC_ROOT / "Metadata" / "lidc-idri-nodule-counts-6-23-2015.xlsx"
    DIAGNOSIS_XLS: Path = LIDC_ROOT / "Metadata" / "tcia-diagnosis-data-2012-04-20.xls"
    
    # Output root - ALL generated files go here
    OUTPUT_ROOT: Path = _env_path(
        "LIDC_OUTPUT",
        LIDC_ROOT / "Output"
    )
    
    # Output subdirectories
    CASES_DIR: Path = OUTPUT_ROOT / "cases"
    NNUNET_ROOT: Path = OUTPUT_ROOT / "nnunet"
    QC_DIR: Path = OUTPUT_ROOT / "qc"
    
    # nnU-Net directories
    NNUNET_RAW: Path = NNUNET_ROOT / "nnUNet_raw"
    NNUNET_PREPROCESSED: Path = NNUNET_ROOT / "nnUNet_preprocessed"
    NNUNET_RESULTS: Path = NNUNET_ROOT / "nnUNet_results"
    
    # Dataset ID for nnU-Net
    NNUNET_DATASET_ID: int = 502
    NNUNET_DATASET_NAME: str = "Dataset502_LIDC"
    
    # Ground truth consensus threshold
    # Using >=2/4 readers as our primary GT definition
    CONSENSUS_THRESHOLD: float = 0.5  # 50% = 2/4 readers
    
    # MedSAM2 configuration (reuse from main project)
    MEDSAM2_ROOT: Path = PROJECT_ROOT / "external" / "MedSAM2"
    MEDSAM2_CHECKPOINT: Path = MEDSAM2_ROOT / "checkpoints" / "MedSAM2_latest.pt"
    MEDSAM2_CONFIG: Path = MEDSAM2_ROOT / "sam2" / "configs" / "sam2.1_hiera_t512.yaml"


@dataclass(frozen=True)
class LIDCCasePaths:
    """Paths for a single LIDC-IDRI case."""
    case_id: str
    case_dir: Path
    
    @property
    def nifti_dir(self) -> Path:
        return self.case_dir / "nifti"
    
    @property
    def seg_dir(self) -> Path:
        return self.case_dir / "seg"
    
    @property
    def qc_dir(self) -> Path:
        return self.case_dir / "qc"
    
    @property
    def manifest_path(self) -> Path:
        return self.case_dir / "manifest.json"
    
    # Standard file paths
    @property
    def ct_nifti(self) -> Path:
        return self.nifti_dir / "ct.nii.gz"
    
    @property
    def nodule_mask_gt(self) -> Path:
        """Ground truth mask from XML annotations."""
        return self.seg_dir / "nodule_mask_gt.nii.gz"
    
    @property
    def nodule_mask_nnunet(self) -> Path:
        """nnU-Net prediction."""
        return self.seg_dir / "nodule_mask_nnunet.nii.gz"
    
    @property
    def nodule_mask_medsam(self) -> Path:
        """MedSAM2 prediction."""
        return self.seg_dir / "nodule_mask_medsam.nii.gz"


def ensure_lidc_dirs(cfg: LIDCConfig) -> None:
    """Create all required output directories."""
    cfg.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cfg.CASES_DIR.mkdir(parents=True, exist_ok=True)
    cfg.QC_DIR.mkdir(parents=True, exist_ok=True)
    cfg.NNUNET_ROOT.mkdir(parents=True, exist_ok=True)
    cfg.NNUNET_RAW.mkdir(parents=True, exist_ok=True)
    cfg.NNUNET_PREPROCESSED.mkdir(parents=True, exist_ok=True)
    cfg.NNUNET_RESULTS.mkdir(parents=True, exist_ok=True)


def ensure_case_dirs(case: LIDCCasePaths) -> None:
    """Create directories for a single case."""
    case.case_dir.mkdir(parents=True, exist_ok=True)
    case.nifti_dir.mkdir(parents=True, exist_ok=True)
    case.seg_dir.mkdir(parents=True, exist_ok=True)
    case.qc_dir.mkdir(parents=True, exist_ok=True)


def get_lidc_case(cfg: LIDCConfig, case_id: str) -> LIDCCasePaths:
    """Get case paths for a LIDC-IDRI patient."""
    return LIDCCasePaths(case_id=case_id, case_dir=cfg.CASES_DIR / case_id)


def list_available_patients(cfg: LIDCConfig) -> list[str]:
    """List all available patient IDs in the DICOM root."""
    if not cfg.DICOM_ROOT.exists():
        return []
    
    patients = []
    for item in sorted(cfg.DICOM_ROOT.iterdir()):
        if item.is_dir() and item.name.startswith("LIDC-IDRI-"):
            patients.append(item.name)
    return patients
