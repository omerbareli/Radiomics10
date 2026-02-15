# pipeline/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tempfile
import uuid


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


def verify_write_access(path: Path) -> None:
    """
    Verify write access to mounted directory. Abort if no access.
    
    Attempts to create a temporary test directory and file under the given path.
    Cleans up test artifacts on success. Raises PermissionError with clear
    message on failure.
    
    Args:
        path: Directory path to verify write access for.
        
    Raises:
        PermissionError: If write access is not available.
    """
    test_id = f".write_test_{uuid.uuid4().hex[:8]}"
    test_dir = path / test_id
    test_file = test_dir / "test_write"
    
    try:
        # Ensure parent exists
        path.mkdir(parents=True, exist_ok=True)
        
        # Try to create test directory
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to write a file
        test_file.write_text("write_access_check")
        
        # Cleanup
        test_file.unlink()
        test_dir.rmdir()
        
    except (PermissionError, OSError) as e:
        # Attempt cleanup even on failure
        try:
            if test_file.exists():
                test_file.unlink()
            if test_dir.exists():
                test_dir.rmdir()
        except Exception:
            pass
        
        raise PermissionError(
            f"No write access to mount point: {path}\n"
            f"Please verify the network drive is mounted and writable.\n"
            f"Original error: {e}"
        ) from e


@dataclass(frozen=True)
class Config:
    # Project root = folder that contains this repo (Radiomics10)
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

    # Raw input lives on /mnt (don't copy it)
    DICOM_ROOT: Path = _env_path(
        "RADIOMICS10_DICOM_ROOT",
        "/mnt/metlab27/D/Input/Radiomics/T_IL Radiomics10/"
        "Lung-PET-CT-Dx/manifest-1608669183333/Lung-PET-CT-Dx",
    )

    ANNOTATION_ROOT: Path = _env_path(
    "RADIOMICS10_ANNOTATION_ROOT",
    "/mnt/metlab27/D/Input/Radiomics/T_IL Radiomics10/Annotation",
    )
    CLINICAL_STATS_XLSX: Path = _env_path(
    "RADIOMICS10_CLINICAL_XLSX",
    "/mnt/metlab27/D/Input/Radiomics/T_IL Radiomics10/statistics-clinical.xlsx",
    )

    # Metadata (keep relative to repo so it works everywhere)
    METADATA_DIR: Path = PROJECT_ROOT / "metadata"
    METADATA_MANIFEST_XLSX: Path = METADATA_DIR / "Lung-PET-CT-Dx-NBIA-Manifest-122220-nbia-digest.xlsx"

    # Data output root - configurable for mounted network drive
    # Set RADIOMICS10_DATA_ROOT environment variable to use a different location
    DATA_ROOT: Path = _env_path(
        "RADIOMICS10_DATA_ROOT",
        "/mnt/radiomics10_data",
    )
    
    # Workspace directories (derived from DATA_ROOT)
    DATA_DIR: Path = DATA_ROOT
    CASES_DIR: Path = DATA_ROOT / "cases"

    # Heuristics
    LOCALIZER_KEYWORDS: tuple[str, ...] = ("localizer", "scout", "topogram", "scanogram")

    # MedSAM2 configuration
    MEDSAM2_ROOT: Path = PROJECT_ROOT / "external" / "MedSAM2"
    MEDSAM2_CHECKPOINT: Path = MEDSAM2_ROOT / "checkpoints" / "MedSAM2_latest.pt"
    MEDSAM2_CONFIG: Path = MEDSAM2_ROOT / "sam2" / "configs" / "sam2.1_hiera_t512.yaml"
    USE_MEDSAM2: bool = False  # Default to bbox mode for backward compatibility


@dataclass(frozen=True)
class CasePaths:
    case_id: str
    case_dir: Path

    @property
    def input_dir(self) -> Path:  # could be symlink to /mnt/.../patient/
        return self.case_dir / "input"

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
    def logs_dir(self) -> Path:
        return self.case_dir / "logs"

    @property
    def manifest_path(self) -> Path:
        return self.case_dir / "manifest.json"

    @property
    def features_dir(self) -> Path:
        return self.case_dir / "features"


def ensure_project_dirs(cfg: Config) -> None:
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg.CASES_DIR.mkdir(parents=True, exist_ok=True)


def ensure_case_dirs(case: CasePaths) -> None:
    case.case_dir.mkdir(parents=True, exist_ok=True)
    case.input_dir.mkdir(parents=True, exist_ok=True)
    case.nifti_dir.mkdir(parents=True, exist_ok=True)
    case.seg_dir.mkdir(parents=True, exist_ok=True)
    case.qc_dir.mkdir(parents=True, exist_ok=True)
    case.logs_dir.mkdir(parents=True, exist_ok=True)
    case.features_dir.mkdir(parents=True, exist_ok=True)


def get_case(cfg: Config, case_id: str) -> CasePaths:
    return CasePaths(case_id=case_id, case_dir=cfg.CASES_DIR / case_id)
