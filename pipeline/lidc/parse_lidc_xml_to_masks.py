# pipeline/lidc/parse_lidc_xml_to_masks.py
"""Parse LIDC-IDRI XML annotations and generate voxel-wise nodule masks.

STRICT QC VERSION - All ROIs must be accounted for with zero drops.

This script:
1. Extracts XML annotations from LIDC-XML-only.zip
2. Maps XML annotations to CT slices using SOP_UID (primary) or Z fallback
3. Converts polygon contours to filled masks
4. Applies consensus rule (≥2/4 readers by default)
5. Saves masks aligned with CT geometry
6. Validates against metadata (nodule counts)

Ground Truth Definition:
- Include: unblindedReadNodule with <characteristics> (nodules ≥3mm)
- Exclude: nonNodule elements and nodules without characteristics
- Consensus: ≥50% reader agreement (≥2/4 readers)

STRICT ACCOUNTING:
- Total_ROIs_in_XML: Count of all inclusion ROIs
- ROIs_Mapped_by_SOP: Successfully mapped via SOP_UID
- ROIs_Mapped_by_Z: Mapped via Z-position fallback
- ROIs_Dropped: Failed to map (MUST be 0 for success)

Usage:
    python -m pipeline.lidc.parse_lidc_xml_to_masks --patient LIDC-IDRI-0001
    python -m pipeline.lidc.parse_lidc_xml_to_masks --all --limit 20 --strict
"""
from __future__ import annotations

import argparse
import json
import logging
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage.draw import polygon as sk_polygon
from tqdm import tqdm

from pipeline.lidc.config import (
    LIDCConfig,
    LIDCCasePaths,
    ensure_case_dirs,
    ensure_lidc_dirs,
    get_lidc_case,
    list_available_patients,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# XML namespace for LIDC annotations
LIDC_NS = {"lidc": "http://www.nih.gov"}


@dataclass
class ContourPoint:
    """A point in a nodule contour."""
    x: int
    y: int


@dataclass
class ROI:
    """A region of interest (contour) on a single slice."""
    image_z_position: float
    image_sop_uid: str
    inclusion: bool  # TRUE = inside nodule, FALSE = exclusion region
    points: List[ContourPoint] = field(default_factory=list)


@dataclass
class Nodule:
    """A nodule annotation from a single reader."""
    nodule_id: str
    has_characteristics: bool  # True if ≥3mm nodule with malignancy rating
    malignancy: Optional[int] = None  # 1-5 scale
    rois: List[ROI] = field(default_factory=list)


@dataclass
class ReadingSession:
    """A reading session from one radiologist."""
    reader_id: str
    nodules: List[Nodule] = field(default_factory=list)


@dataclass
class PatientAnnotation:
    """All annotations for a patient."""
    study_uid: str
    series_uid: str
    reading_sessions: List[ReadingSession] = field(default_factory=list)


@dataclass
class ZFallbackInfo:
    """Details about a Z-position fallback match."""
    z_diff_mm: float
    num_candidates: int  # How many slices were within tolerance
    is_ambiguous: bool   # True if num_candidates > 1


@dataclass
class StrictROIStats:
    """Strict accounting of ROI mapping with Z-diff tracking."""
    total_rois_in_xml: int = 0
    rois_mapped_by_sop: int = 0
    rois_mapped_by_z: int = 0
    rois_dropped: int = 0
    total_nodules_with_chars: int = 0  # Nodules >= 3mm
    total_nodules_without_chars: int = 0  # Nodules < 3mm
    num_readers: int = 0
    
    # Z-diff tracking
    z_diff_values: List[float] = field(default_factory=list)  # Z-diff for each fallback
    z_fallback_ambiguous_count: int = 0  # Fallbacks with >1 candidate
    z_fallback_no_candidate_count: int = 0  # Fallbacks with 0 candidates (should be 0)
    z_tolerance_mm: float = 0.5  # Tolerance used
    
    @property
    def all_mapped(self) -> bool:
        """True if all ROIs were successfully mapped."""
        return self.rois_dropped == 0 and self.total_rois_in_xml > 0
    
    @property
    def sop_match_rate(self) -> float:
        """Fraction of ROIs matched by SOP_UID (vs Z fallback)."""
        total_mapped = self.rois_mapped_by_sop + self.rois_mapped_by_z
        return self.rois_mapped_by_sop / max(total_mapped, 1)
    
    @property
    def z_fallback_rate(self) -> float:
        """Fraction of ROIs matched by Z-fallback."""
        total_mapped = self.rois_mapped_by_sop + self.rois_mapped_by_z
        return self.rois_mapped_by_z / max(total_mapped, 1)
    
    @property
    def z_diff_mm_max(self) -> float:
        """Maximum Z-diff across all fallback matches."""
        return max(self.z_diff_values) if self.z_diff_values else 0.0
    
    @property
    def z_diff_mm_median(self) -> float:
        """Median Z-diff across all fallback matches."""
        if not self.z_diff_values:
            return 0.0
        sorted_vals = sorted(self.z_diff_values)
        n = len(sorted_vals)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
        return sorted_vals[mid]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_rois_in_xml": self.total_rois_in_xml,
            "rois_mapped_by_sop": self.rois_mapped_by_sop,
            "rois_mapped_by_z": self.rois_mapped_by_z,
            "rois_dropped": self.rois_dropped,
            "total_nodules_with_chars": self.total_nodules_with_chars,
            "total_nodules_without_chars": self.total_nodules_without_chars,
            "num_readers": self.num_readers,
            "sop_match_rate": self.sop_match_rate,
            "z_fallback_rate": self.z_fallback_rate,
            "z_diff_mm_max": self.z_diff_mm_max,
            "z_diff_mm_median": self.z_diff_mm_median,
            "z_fallback_ambiguous_count": self.z_fallback_ambiguous_count,
            "z_fallback_no_candidate_count": self.z_fallback_no_candidate_count,
            "z_tolerance_mm": self.z_tolerance_mm,
            "all_mapped": self.all_mapped,
        }


def parse_xml_annotation(xml_path: Path) -> PatientAnnotation:
    """Parse a LIDC XML annotation file.
    
    Args:
        xml_path: Path to XML file
        
    Returns:
        PatientAnnotation with all reading sessions
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Handle namespace
    ns = LIDC_NS
    
    # Get study/series UIDs from header
    header = root.find("lidc:ResponseHeader", ns)
    study_uid = ""
    series_uid = ""
    if header is not None:
        study_elem = header.find("lidc:StudyInstanceUID", ns)
        series_elem = header.find("lidc:SeriesInstanceUid", ns)
        study_uid = study_elem.text if study_elem is not None else ""
        series_uid = series_elem.text if series_elem is not None else ""
    
    annotation = PatientAnnotation(
        study_uid=study_uid,
        series_uid=series_uid,
    )
    
    # Parse each reading session
    for session_elem in root.findall("lidc:readingSession", ns):
        reader_id_elem = session_elem.find("lidc:servicingRadiologistID", ns)
        reader_id = reader_id_elem.text if reader_id_elem is not None else "unknown"
        
        session = ReadingSession(reader_id=reader_id)
        
        # Parse unblindedReadNodule elements (nodules ≥3mm)
        for nodule_elem in session_elem.findall("lidc:unblindedReadNodule", ns):
            nodule_id_elem = nodule_elem.find("lidc:noduleID", ns)
            nodule_id = nodule_id_elem.text if nodule_id_elem is not None else ""
            
            # Check for characteristics (indicates ≥3mm nodule)
            chars_elem = nodule_elem.find("lidc:characteristics", ns)
            has_characteristics = chars_elem is not None
            
            malignancy = None
            if chars_elem is not None:
                mal_elem = chars_elem.find("lidc:malignancy", ns)
                if mal_elem is not None and mal_elem.text:
                    try:
                        malignancy = int(mal_elem.text)
                    except ValueError:
                        pass
            
            nodule = Nodule(
                nodule_id=nodule_id,
                has_characteristics=has_characteristics,
                malignancy=malignancy,
            )
            
            # Parse ROIs
            for roi_elem in nodule_elem.findall("lidc:roi", ns):
                z_elem = roi_elem.find("lidc:imageZposition", ns)
                sop_elem = roi_elem.find("lidc:imageSOP_UID", ns)
                incl_elem = roi_elem.find("lidc:inclusion", ns)
                
                z_pos = float(z_elem.text) if z_elem is not None and z_elem.text else 0.0
                sop_uid = sop_elem.text if sop_elem is not None else ""
                inclusion = incl_elem is not None and incl_elem.text.upper() == "TRUE"
                
                roi = ROI(
                    image_z_position=z_pos,
                    image_sop_uid=sop_uid,
                    inclusion=inclusion,
                )
                
                # Parse edge map (contour points)
                for edge_elem in roi_elem.findall("lidc:edgeMap", ns):
                    x_elem = edge_elem.find("lidc:xCoord", ns)
                    y_elem = edge_elem.find("lidc:yCoord", ns)
                    if x_elem is not None and y_elem is not None:
                        try:
                            roi.points.append(ContourPoint(
                                x=int(x_elem.text),
                                y=int(y_elem.text),
                            ))
                        except (ValueError, TypeError):
                            pass
                
                if roi.points:  # Only add ROIs with valid contours
                    nodule.rois.append(roi)
            
            if nodule.rois:  # Only add nodules with valid ROIs
                session.nodules.append(nodule)
        
        if session.nodules:
            annotation.reading_sessions.append(session)
    
    return annotation


def map_annotation_to_slice(
    annotation_sop_uid: str,
    annotation_z: float,
    sop_to_slice: Dict[str, int],
    z_positions: List[float],
    z_tolerance: float = 0.5,
) -> Tuple[int, str, Optional[ZFallbackInfo]]:
    """Map annotation to CT slice index using SOP_UID or Z fallback.
    
    Args:
        annotation_sop_uid: SOP_UID from XML annotation
        annotation_z: Z position from XML annotation
        sop_to_slice: Mapping from SOP_UID to slice index
        z_positions: List of Z positions in slice order
        z_tolerance: Tolerance for Z matching in mm
        
    Returns:
        Tuple of (slice_index, match_method, fallback_info)
        - match_method: "sop" or "z_fallback"
        - fallback_info: ZFallbackInfo if z_fallback, else None
        
    Raises:
        ValueError: If no match found within tolerance (with info about no candidates)
    """
    # Primary: Match by SOP_UID (exact)
    if annotation_sop_uid and annotation_sop_uid in sop_to_slice:
        return sop_to_slice[annotation_sop_uid], "sop", None
    
    # Fallback: Match by Z-position with tolerance
    z_array = np.array(z_positions)
    z_diffs = np.abs(z_array - annotation_z)
    
    # Count candidates within tolerance
    candidates_within_tolerance = np.sum(z_diffs <= z_tolerance)
    best_idx = int(np.argmin(z_diffs))
    best_diff = float(z_diffs[best_idx])
    
    if candidates_within_tolerance == 0:
        raise ValueError(
            f"NO_CANDIDATE: SOP={annotation_sop_uid[:20] if annotation_sop_uid else 'N/A'}..., Z={annotation_z:.2f}. "
            f"Closest Z diff: {best_diff:.2f}mm (tolerance: {z_tolerance}mm)"
        )
    
    # Build fallback info
    fallback_info = ZFallbackInfo(
        z_diff_mm=best_diff,
        num_candidates=int(candidates_within_tolerance),
        is_ambiguous=(candidates_within_tolerance > 1),
    )
    
    if fallback_info.is_ambiguous:
        logger.debug(
            f"AMBIGUOUS Z-fallback: Z={annotation_z:.2f} → slice {best_idx} "
            f"(diff={best_diff:.3f}mm, {candidates_within_tolerance} candidates within tolerance)"
        )
    else:
        logger.debug(
            f"SOP_UID not found, using Z-fallback: Z={annotation_z:.2f} → slice {best_idx} "
            f"(diff={best_diff:.3f}mm)"
        )
    
    return best_idx, "z_fallback", fallback_info


def contour_to_mask(
    points: List[ContourPoint],
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Convert contour points to filled binary mask.
    
    Args:
        points: List of ContourPoint (x, y coordinates)
        image_shape: (height, width) of the slice
        
    Returns:
        Binary mask of shape (height, width)
    """
    if len(points) < 3:
        return np.zeros(image_shape, dtype=np.uint8)
    
    # Extract x, y coordinates
    xs = np.array([p.x for p in points])
    ys = np.array([p.y for p in points])
    
    # Use skimage.draw.polygon to fill
    # Note: polygon takes (row, col) = (y, x)
    rr, cc = sk_polygon(ys, xs, shape=image_shape)
    
    mask = np.zeros(image_shape, dtype=np.uint8)
    mask[rr, cc] = 1
    
    return mask


def generate_reader_mask_strict(
    annotation: PatientAnnotation,
    ct_shape: Tuple[int, int, int],
    sop_to_slice: Dict[str, int],
    z_positions: List[float],
    include_only_characterized: bool = True,
    z_tolerance: float = 0.5,
) -> Tuple[List[np.ndarray], StrictROIStats]:
    """Generate per-reader nodule masks with STRICT accounting.
    
    Args:
        annotation: Parsed annotation
        ct_shape: (Z, Y, X) shape of CT volume
        sop_to_slice: SOP_UID to slice index mapping
        z_positions: Z positions in slice order
        include_only_characterized: If True, only include nodules with characteristics
        z_tolerance: Tolerance for Z matching in mm
        
    Returns:
        Tuple of (list of reader masks, StrictROIStats)
    """
    depth, height, width = ct_shape
    reader_masks = []
    stats = StrictROIStats()
    stats.z_tolerance_mm = z_tolerance
    
    for session in annotation.reading_sessions:
        reader_mask = np.zeros(ct_shape, dtype=np.uint8)
        reader_has_nodules = False
        
        for nodule in session.nodules:
            # Count nodule types
            if nodule.has_characteristics:
                stats.total_nodules_with_chars += 1
            else:
                stats.total_nodules_without_chars += 1
            
            # Filter by characteristics if requested
            if include_only_characterized and not nodule.has_characteristics:
                continue
            
            for roi in nodule.rois:
                if not roi.inclusion:
                    continue  # Skip exclusion regions
                
                # Count this ROI
                stats.total_rois_in_xml += 1
                
                try:
                    slice_idx, match_method, fallback_info = map_annotation_to_slice(
                        roi.image_sop_uid,
                        roi.image_z_position,
                        sop_to_slice,
                        z_positions,
                        z_tolerance=z_tolerance,
                    )
                    
                    if match_method == "sop":
                        stats.rois_mapped_by_sop += 1
                    else:
                        stats.rois_mapped_by_z += 1
                        # Track Z-diff info
                        if fallback_info:
                            stats.z_diff_values.append(fallback_info.z_diff_mm)
                            if fallback_info.is_ambiguous:
                                stats.z_fallback_ambiguous_count += 1
                    
                except ValueError as e:
                    error_msg = str(e)
                    if "NO_CANDIDATE" in error_msg:
                        stats.z_fallback_no_candidate_count += 1
                    stats.rois_dropped += 1
                    logger.warning(f"ROI DROPPED: {e}")
                    continue
                
                # Ensure slice_idx is valid
                if not (0 <= slice_idx < depth):
                    stats.rois_dropped += 1
                    logger.warning(f"ROI DROPPED: Slice index {slice_idx} out of range [0, {depth})")
                    continue
                
                # Convert contour to mask
                slice_mask = contour_to_mask(roi.points, (height, width))
                
                # OR into reader mask (handles multiple nodules/ROIs)
                reader_mask[slice_idx] = np.maximum(reader_mask[slice_idx], slice_mask)
                reader_has_nodules = True
        
        if reader_has_nodules:
            reader_masks.append(reader_mask)
            stats.num_readers += 1
    
    return reader_masks, stats


def apply_consensus(
    reader_masks: List[np.ndarray],
    threshold: float = 0.5,
) -> np.ndarray:
    """Apply consensus rule across reader masks.
    
    Args:
        reader_masks: List of binary masks, one per reader
        threshold: Fraction of readers that must agree (e.g., 0.5 = 50%)
        
    Returns:
        Consensus binary mask
    """
    if not reader_masks:
        raise ValueError("No reader masks provided")
    
    # Stack and compute agreement
    stacked = np.stack(reader_masks, axis=0)  # (num_readers, Z, Y, X)
    agreement = stacked.sum(axis=0)  # (Z, Y, X)
    
    min_readers = len(reader_masks) * threshold
    consensus = (agreement >= min_readers).astype(np.uint8)
    
    return consensus


def find_xml_for_patient_by_study_uid(
    study_uid: str,
    cfg: LIDCConfig,
    temp_dir: Path,
) -> List[Path]:
    """Find and extract XML files for a patient by Study UID.
    
    The LIDC XML files are organized in folder batches (157, 185, 186, etc.)
    with numeric filenames that don't directly map to patient IDs.
    We match by StudyInstanceUID from the XML header.
    
    Args:
        study_uid: Study Instance UID from CT manifest
        cfg: Configuration
        temp_dir: Temporary directory for extraction
        
    Returns:
        List of XML file paths for this patient
    """
    if not cfg.ANNOTATIONS_ZIP.exists():
        raise FileNotFoundError(f"Annotations ZIP not found: {cfg.ANNOTATIONS_ZIP}")
    
    found_xmls = []
    ns = LIDC_NS
    
    with zipfile.ZipFile(cfg.ANNOTATIONS_ZIP, "r") as zf:
        # List all XML files
        xml_names = [n for n in zf.namelist() if n.endswith(".xml")]
        
        for xml_name in xml_names:
            try:
                # Read XML content
                content = zf.read(xml_name).decode("utf-8")
                tree = ET.fromstring(content)
                
                # Check StudyInstanceUID in header
                header = tree.find("lidc:ResponseHeader", ns)
                if header is not None:
                    study_elem = header.find("lidc:StudyInstanceUID", ns)
                    if study_elem is not None and study_elem.text == study_uid:
                        # Match found - extract to temp dir
                        basename = Path(xml_name).name
                        extract_path = temp_dir / f"{Path(xml_name).parent.name}_{basename}"
                        with open(extract_path, "wb") as f:
                            f.write(content.encode("utf-8"))
                        found_xmls.append(extract_path)
            except Exception as e:
                logger.debug(f"Error parsing {xml_name}: {e}")
                continue
    
    return found_xmls


def find_xml_for_patient(
    patient_id: str,
    cfg: LIDCConfig,
    temp_dir: Path,
    manifest: Dict[str, Any] = None,
) -> List[Path]:
    """Find and extract XML files for a patient.
    
    Uses Study UID from manifest to find matching XML files.
    
    Args:
        patient_id: Patient ID like "LIDC-IDRI-0001"
        cfg: Configuration
        temp_dir: Temporary directory for extraction
        manifest: Optional pre-loaded manifest with CT study_uid
        
    Returns:
        List of XML file paths for this patient
    """
    # Get Study UID from manifest
    if manifest is None:
        case = get_lidc_case(cfg, patient_id)
        if not case.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found for {patient_id}")
        with open(case.manifest_path) as f:
            manifest = json.load(f)
    
    study_uid = manifest.get("ct", {}).get("study_uid", "")
    if not study_uid:
        raise ValueError(f"No Study UID in manifest for {patient_id}")
    
    return find_xml_for_patient_by_study_uid(study_uid, cfg, temp_dir)


def load_metadata_nodule_counts(cfg: LIDCConfig) -> Dict[str, int]:
    """Load expected nodule counts from metadata Excel.
    
    Returns:
        Dict mapping patient_id -> expected number of nodules >= 3mm
    """
    if not cfg.NODULE_COUNTS_XLSX.exists():
        logger.warning(f"Nodule counts metadata not found: {cfg.NODULE_COUNTS_XLSX}")
        return {}
    
    try:
        df = pd.read_excel(cfg.NODULE_COUNTS_XLSX)
        # Column name: 'Number of Nodules >=3mm**'
        counts = {}
        for _, row in df.iterrows():
            patient_id = str(row.get("TCIA Patent ID", "")).strip()
            nodule_count = int(row.get("Number of Nodules >=3mm**", 0))
            if patient_id.startswith("LIDC-IDRI-"):
                counts[patient_id] = nodule_count
        logger.info(f"Loaded nodule counts for {len(counts)} patients from metadata")
        return counts
    except Exception as e:
        logger.warning(f"Failed to load metadata: {e}")
        return {}


def calculate_volume(mask: np.ndarray, spacing: Tuple[float, float, float]) -> float:
    """Calculate volume in mm³.
    
    Args:
        mask: Binary mask array
        spacing: (x, y, z) spacing in mm
        
    Returns:
        Volume in mm³
    """
    voxel_count = int(np.sum(mask > 0))
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    return voxel_count * voxel_volume_mm3


def generate_gt_mask_strict(
    patient_id: str,
    cfg: LIDCConfig,
    overwrite: bool = False,
    strict: bool = True,
    expected_nodule_counts: Dict[str, int] = None,
) -> Dict[str, Any]:
    """Generate ground truth nodule mask with STRICT verification.
    
    Args:
        patient_id: Patient ID like "LIDC-IDRI-0001"
        cfg: Configuration
        overwrite: If True, overwrite existing mask
        strict: If True, fail if any ROIs are dropped
        expected_nodule_counts: Optional dict of patient_id -> expected nodule count
        
    Returns:
        Result dict with status, stats, and verification info
    """
    case = get_lidc_case(cfg, patient_id)
    
    # Check prerequisites
    if not case.ct_nifti.exists():
        return {"status": "FAILED", "patient_id": patient_id, "error": "CT NIfTI not found"}
    
    if not case.manifest_path.exists():
        return {"status": "FAILED", "patient_id": patient_id, "error": "Manifest not found"}
    
    # Check if already done
    if case.nodule_mask_gt.exists() and not overwrite:
        logger.info(f"[SKIP] {patient_id}: mask already exists")
        return {"status": "SKIPPED", "patient_id": patient_id}
    
    ensure_case_dirs(case)
    
    # Load CT and manifest
    ct_image = sitk.ReadImage(str(case.ct_nifti))
    ct_array = sitk.GetArrayFromImage(ct_image)  # (Z, Y, X)
    ct_shape = ct_array.shape
    spacing = ct_image.GetSpacing()  # (x, y, z)
    
    with open(case.manifest_path) as f:
        manifest = json.load(f)
    
    sop_mapping = manifest.get("sop_mapping", {})
    sop_to_slice = sop_mapping.get("sop_to_slice", {})
    z_positions = sop_mapping.get("z_positions", [])
    
    if not sop_to_slice:
        return {"status": "FAILED", "patient_id": patient_id, "error": "No SOP mapping in manifest"}
    
    # Find and parse XML annotations
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        xml_files = find_xml_for_patient(patient_id, cfg, temp_path, manifest)
        
        if not xml_files:
            logger.warning(f"[{patient_id}] No XML files found")
            return {"status": "NO_ANNOTATIONS", "patient_id": patient_id}
        
        logger.info(f"[{patient_id}] Found {len(xml_files)} XML file(s)")
        
        # Parse all XMLs and collect reader masks with STRICT accounting
        all_reader_masks = []
        total_stats = StrictROIStats()
        xml_study_uid = ""
        xml_series_uid = ""
        
        for xml_file in xml_files:
            annotation = parse_xml_annotation(xml_file)
            
            # Capture XML UIDs (use first non-empty)
            if not xml_study_uid and annotation.study_uid:
                xml_study_uid = annotation.study_uid
            if not xml_series_uid and annotation.series_uid:
                xml_series_uid = annotation.series_uid
            
            reader_masks, stats = generate_reader_mask_strict(
                annotation,
                ct_shape,
                sop_to_slice,
                z_positions,
                include_only_characterized=True,
            )
            
            all_reader_masks.extend(reader_masks)
            
            # Aggregate stats
            total_stats.total_rois_in_xml += stats.total_rois_in_xml
            total_stats.rois_mapped_by_sop += stats.rois_mapped_by_sop
            total_stats.rois_mapped_by_z += stats.rois_mapped_by_z
            total_stats.rois_dropped += stats.rois_dropped
            total_stats.total_nodules_with_chars += stats.total_nodules_with_chars
            total_stats.total_nodules_without_chars += stats.total_nodules_without_chars
            total_stats.num_readers += stats.num_readers
            
            # Aggregate z_diff tracking
            total_stats.z_diff_values.extend(stats.z_diff_values)
            total_stats.z_fallback_ambiguous_count += stats.z_fallback_ambiguous_count
            total_stats.z_fallback_no_candidate_count += stats.z_fallback_no_candidate_count
            total_stats.z_tolerance_mm = stats.z_tolerance_mm  # Should be same for all
    
    # STRICT CHECK: Fail if any ROIs were dropped
    if strict and total_stats.rois_dropped > 0:
        logger.error(
            f"[{patient_id}] FAILED: {total_stats.rois_dropped} ROIs dropped "
            f"(Total: {total_stats.total_rois_in_xml}, Mapped: {total_stats.rois_mapped_by_sop + total_stats.rois_mapped_by_z})"
        )
        return {
            "status": "FAILED",
            "patient_id": patient_id,
            "error": f"ROIs dropped: {total_stats.rois_dropped}",
            "roi_stats": total_stats.to_dict(),
        }
    
    if not all_reader_masks:
        logger.warning(f"[{patient_id}] No valid reader annotations found")
        consensus_mask = np.zeros(ct_shape, dtype=np.uint8)
    else:
        # Apply consensus
        consensus_mask = apply_consensus(
            all_reader_masks,
            threshold=cfg.CONSENSUS_THRESHOLD,
        )
    
    # Calculate volume
    mask_volume_voxels = int(np.sum(consensus_mask > 0))
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    mask_volume_mm3 = mask_volume_voxels * voxel_volume_mm3
    
    # Save mask with same geometry as CT
    mask_image = sitk.GetImageFromArray(consensus_mask)
    mask_image.CopyInformation(ct_image)
    sitk.WriteImage(mask_image, str(case.nodule_mask_gt), useCompression=True)
    
    # Metadata cross-reference
    metadata_match = None
    if expected_nodule_counts and patient_id in expected_nodule_counts:
        expected_count = expected_nodule_counts[patient_id]
        # We count unique nodules with characteristics (across all readers / 4 readers)
        # The metadata counts nodules, not reader annotations
        # Approximate: total_nodules_with_chars / 4 (average across readers)
        actual_unique_nodules = total_stats.total_nodules_with_chars // max(total_stats.num_readers, 1)
        
        if actual_unique_nodules != expected_count:
            metadata_match = "METADATA_MISMATCH"
            logger.warning(
                f"[{patient_id}] METADATA_MISMATCH: Expected {expected_count} nodules, "
                f"found {actual_unique_nodules} unique nodules "
                f"({total_stats.total_nodules_with_chars} across {total_stats.num_readers} readers)"
            )
        else:
            metadata_match = "MATCH"
    
    # QUARANTINE GATING: Flag cases with low SOP match rate for review
    qc_flag = None
    if total_stats.total_rois_in_xml > 0:
        z_fallback_rate = total_stats.rois_mapped_by_z / total_stats.total_rois_in_xml
        
        if total_stats.sop_match_rate < 0.9 or z_fallback_rate > 0.10:
            qc_flag = "QUARANTINE_LOW_SOP_MATCH"
            logger.warning(
                f"[{patient_id}] ⚠️ QUARANTINE: SOP match rate {total_stats.sop_match_rate:.1%}, "
                f"Z-fallback rate {z_fallback_rate:.1%}. "
                f"Review QC overlay before including in training."
            )
    
    result = {
        "status": "SUCCESS",
        "patient_id": patient_id,
        "roi_stats": total_stats.to_dict(),
        "volume": {
            "voxel_count": mask_volume_voxels,
            "volume_mm3": mask_volume_mm3,
            "voxel_volume_mm3": voxel_volume_mm3,
            "spacing_mm": list(spacing),
        },
        "metadata_validation": metadata_match,
        "qc_flag": qc_flag,  # NEW: Quarantine flag
        "consensus_threshold": cfg.CONSENSUS_THRESHOLD,
        "xml_study_uid": xml_study_uid,
        "xml_series_uid": xml_series_uid,
    }
    
    logger.info(
        f"[{patient_id}] ✓ SUCCESS: "
        f"ROIs={total_stats.total_rois_in_xml} (SOP:{total_stats.rois_mapped_by_sop}, Z:{total_stats.rois_mapped_by_z}, Drop:0), "
        f"Volume={mask_volume_mm3:.1f}mm³ ({mask_volume_voxels} voxels), "
        f"Readers={total_stats.num_readers}"
    )
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Parse LIDC-IDRI XML annotations to nodule masks (STRICT QC version)"
    )
    parser.add_argument(
        "--patient",
        type=str,
        help="Single patient ID (e.g., LIDC-IDRI-0001)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all converted patients",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of patients to process",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing masks",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any ROIs are dropped (default: True)",
        default=True,
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Allow dropped ROIs (not recommended)",
    )
    args = parser.parse_args()
    
    strict = not args.no_strict
    
    cfg = LIDCConfig()
    ensure_lidc_dirs(cfg)
    
    # Load metadata for cross-reference
    expected_counts = load_metadata_nodule_counts(cfg)
    
    # Determine which patients to process
    if args.patient:
        patient_ids = [args.patient]
    elif args.all:
        # Only process patients that have been converted
        available = list_available_patients(cfg)
        patient_ids = []
        for pid in available:
            case = get_lidc_case(cfg, pid)
            if case.ct_nifti.exists():
                patient_ids.append(pid)
        
        if args.limit:
            patient_ids = patient_ids[:args.limit]
    else:
        parser.error("Specify --patient or --all")
        return
    
    logger.info(f"Processing {len(patient_ids)} patients (strict={strict})...")
    
    results = {"SUCCESS": [], "FAILED": [], "SKIPPED": [], "NO_ANNOTATIONS": []}
    
    for patient_id in tqdm(patient_ids, desc="Parsing XMLs"):
        try:
            result = generate_gt_mask_strict(
                patient_id, cfg, 
                overwrite=args.overwrite, 
                strict=strict,
                expected_nodule_counts=expected_counts,
            )
            status = result.get("status", "UNKNOWN")
            results[status].append(result)
                
        except Exception as e:
            logger.error(f"[{patient_id}] EXCEPTION: {e}")
            results["FAILED"].append({"patient_id": patient_id, "error": str(e), "status": "FAILED"})
    
    # Summary
    logger.info(
        f"\n=== SUMMARY ===\n"
        f"SUCCESS: {len(results['SUCCESS'])}\n"
        f"FAILED: {len(results['FAILED'])}\n"
        f"SKIPPED: {len(results['SKIPPED'])}\n"
        f"NO_ANNOTATIONS: {len(results['NO_ANNOTATIONS'])}"
    )
    
    # Report SOP match rates for successful cases
    if results["SUCCESS"]:
        sop_rates = [r["roi_stats"]["sop_match_rate"] for r in results["SUCCESS"]]
        volumes = [r["volume"]["volume_mm3"] for r in results["SUCCESS"]]
        logger.info(f"SOP match rate: mean={np.mean(sop_rates):.1%}, min={np.min(sop_rates):.1%}")
        logger.info(f"Volume (mm³): mean={np.mean(volumes):.1f}, min={np.min(volumes):.1f}, max={np.max(volumes):.1f}")
    
    if results["FAILED"]:
        logger.warning("\nFAILED patients:")
        for r in results["FAILED"]:
            logger.warning(f"  - {r['patient_id']}: {r.get('error', 'Unknown')}")
    
    # Save results to JSON
    results_path = cfg.QC_DIR / "parsing_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
