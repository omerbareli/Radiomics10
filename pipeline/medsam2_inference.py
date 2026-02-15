# pipeline/medsam2_inference.py
"""
MedSAM2 Inference Wrapper for lesion segmentation.

This module provides a wrapper around MedSAM2 for segmenting lesions
in 3D CT volumes using bounding box prompts.

FIXED ISSUES:
- Mask resizing: 512x512 predictions are resampled to original (Y, X) dimensions
- Boolean indexing: Properly using NumPy assignment for 2D masks
- Autocast: Only enabled on CUDA when available
- Config path: Now respects user-provided config_path
- Performance: Crop-based inference for faster multi-bbox processing
- LCC: Per-lesion LCC, with option to preserve multiple lesions
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# Add MedSAM2 to path if needed
MEDSAM2_ROOT = Path(__file__).resolve().parents[1] / "external" / "MedSAM2"
if str(MEDSAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(MEDSAM2_ROOT))

from sam2.build_sam import build_sam2_video_predictor_npz
from skimage import measure
from scipy import ndimage


# Default paths
DEFAULT_CHECKPOINT = MEDSAM2_ROOT / "checkpoints" / "MedSAM2_latest.pt"
DEFAULT_CONFIG = MEDSAM2_ROOT / "sam2" / "configs" / "sam2.1_hiera_t512.yaml"

# Model input size (MedSAM2 expects 512x512)
MODEL_SIZE = 512


@contextmanager
def _autocast_context(device: str, dtype: torch.dtype = torch.float16):
    """
    Context manager for autocast that handles CPU/GPU gracefully.
    Only uses CUDA autocast when on CUDA device.
    """
    use_cuda_autocast = device.startswith("cuda") and torch.cuda.is_available()
    
    if use_cuda_autocast:
        with torch.autocast("cuda", dtype=dtype):
            yield
    else:
        # No autocast on CPU - just yield (no-op context)
        yield


def _resize_mask_to_original(
    mask_512: np.ndarray,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    """
    Resize a 512x512 mask back to original dimensions using nearest neighbor.
    
    Args:
        mask_512: Binary mask of shape (512, 512)
        target_height: Original height (Y dimension)
        target_width: Original width (X dimension)
        
    Returns:
        Resized mask of shape (target_height, target_width)
    """
    if mask_512.shape == (target_height, target_width):
        return mask_512
    
    # Use PIL for consistent nearest-neighbor resizing
    mask_pil = Image.fromarray(mask_512.astype(np.uint8) * 255)
    mask_resized = mask_pil.resize((target_width, target_height), Image.NEAREST)
    return (np.array(mask_resized) > 127).astype(np.uint8)


def _remove_small_components(
    mask: np.ndarray,
    min_voxels: int = 50,
) -> np.ndarray:
    """
    Remove small connected components below a voxel threshold.
    Preserves multiple lesions if they exceed the threshold.
    
    Args:
        mask: Binary 3D mask
        min_voxels: Minimum number of voxels to keep a component
        
    Returns:
        Cleaned mask with small components removed
    """
    if np.max(mask) == 0:
        return mask
    
    labels, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask
    
    # Count voxels per component
    component_sizes = ndimage.sum(mask, labels, range(1, num_features + 1))
    
    # Create output mask keeping only large enough components
    result = np.zeros_like(mask)
    for i, size in enumerate(component_sizes, start=1):
        if size >= min_voxels:
            result[labels == i] = 1
    
    return result


class MedSAM2Segmenter:
    """
    Wrapper for MedSAM2 to segment lesions in 3D CT volumes.
    
    Features:
    - Crop-based inference for performance (only processes region around lesion)
    - Proper mask resizing from 512x512 to original CT dimensions
    - CPU/GPU compatible autocast handling
    - Multi-lesion preservation (optional LCC per lesion, not global)
    
    Example usage:
        segmenter = MedSAM2Segmenter()
        mask = segmenter.segment_lesion_from_bbox(
            ct_volume=ct_array,  # (Z, Y, X)
            key_slice_idx=50,
            bbox=(100, 100, 200, 200),  # (xmin, ymin, xmax, ymax)
        )
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize MedSAM2 predictor.
        
        Args:
            checkpoint_path: Path to MedSAM2 checkpoint (.pt file)
            config_path: Path to model config YAML (relative to sam2 package or absolute)
            device: Device to use ("cuda" or "cpu"). Auto-detects if None.
        """
        self.checkpoint = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
        self.config = Path(config_path) if config_path else DEFAULT_CONFIG
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"MedSAM2 checkpoint not found: {self.checkpoint}")
        if not self.config.exists():
            raise FileNotFoundError(f"MedSAM2 config not found: {self.config}")
        
        # Set torch precision and seeds for reproducibility
        torch.set_float32_matmul_precision('high')
        torch.manual_seed(2024)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(2024)
        np.random.seed(2024)
        
        # Build predictor (lazy loading - builds on first use)
        self._predictor = None
        
        # Cache for preprocessed volumes to avoid re-encoding
        self._volume_cache = None
        self._volume_cache_key = None
    
    def _get_config_relative_path(self) -> str:
        """
        Get the config path relative to sam2 package for hydra.
        
        Hydra's initialize_config_module("sam2") expects configs in sam2/configs/.
        So we need "configs/sam2.1_hiera_t512.yaml" format.
        """
        # If config is under sam2/configs/, extract relative path
        config_str = str(self.config)
        if "sam2/configs/" in config_str:
            # Extract path relative to sam2 package
            idx = config_str.find("sam2/configs/")
            return config_str[idx + len("sam2/"):]  # "configs/..."
        elif "configs/" in config_str:
            # Already relative
            idx = config_str.find("configs/")
            return config_str[idx:]
        else:
            # Use filename only in configs/
            return f"configs/{self.config.name}"
    
    @property
    def predictor(self):
        """Lazy-load the predictor on first access."""
        if self._predictor is None:
            # MedSAM2/hydra expects to be run from MedSAM2 directory with relative config path
            old_cwd = os.getcwd()
            try:
                os.chdir(MEDSAM2_ROOT)
                config_rel = self._get_config_relative_path()
                self._predictor = build_sam2_video_predictor_npz(
                    config_rel, 
                    str(self.checkpoint),
                    device=self.device,
                )
            finally:
                os.chdir(old_cwd)
        return self._predictor
    
    def _preprocess_ct_volume(
        self,
        ct_volume: np.ndarray,
        window_lower: float = -1000,
        window_upper: float = 400,
    ) -> np.ndarray:
        """
        Preprocess CT volume: apply windowing and normalize to 0-255.
        
        Args:
            ct_volume: 3D CT volume in HU (Z, Y, X)
            window_lower: Lower bound of CT window (HU)
            window_upper: Upper bound of CT window (HU)
            
        Returns:
            Preprocessed volume as uint8 (0-255)
        """
        # Clip to window
        clipped = np.clip(ct_volume, window_lower, window_upper)
        
        # Normalize to 0-255
        normalized = (clipped - window_lower) / (window_upper - window_lower) * 255.0
        return normalized.astype(np.uint8)
    
    def _resize_to_rgb(
        self,
        volume: np.ndarray,
        target_size: int = MODEL_SIZE,
    ) -> torch.Tensor:
        """
        Convert grayscale volume to RGB and resize for MedSAM2.
        
        Args:
            volume: 3D uint8 volume (Z, Y, X)
            target_size: Target size for resizing (default 512)
            
        Returns:
            Tensor of shape (Z, 3, target_size, target_size)
        """
        d, h, w = volume.shape
        resized = np.zeros((d, 3, target_size, target_size), dtype=np.float32)
        
        for i in range(d):
            img_pil = Image.fromarray(volume[i])
            img_rgb = img_pil.convert("RGB")
            img_resized = img_rgb.resize((target_size, target_size))
            img_array = np.array(img_resized).transpose(2, 0, 1)  # (3, H, W)
            resized[i] = img_array
        
        # Normalize with ImageNet stats
        tensor = torch.from_numpy(resized).to(self.device)
        tensor = tensor / 255.0
        
        img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device)[:, None, None]
        img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device)[:, None, None]
        tensor = (tensor - img_mean) / img_std
        
        return tensor
    
    def _get_largest_connected_component(self, segmentation: np.ndarray) -> np.ndarray:
        """Keep only the largest connected component in a 3D mask."""
        if np.max(segmentation) == 0:
            return segmentation
        labels = measure.label(segmentation)
        counts = np.bincount(labels.flat)
        # Skip background (label 0)
        if len(counts) <= 1:
            return segmentation
        largest_label = np.argmax(counts[1:]) + 1
        return (labels == largest_label).astype(np.uint8)
    
    def _compute_crop_bounds(
        self,
        bbox: Tuple[int, int, int, int],
        key_slice_idx: int,
        volume_shape: Tuple[int, int, int],
        margin_xy: int = 32,
        margin_z: int = 10,
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Compute crop bounds around a bbox with margins.
        
        Args:
            bbox: (xmin, ymin, xmax, ymax) in original coords
            key_slice_idx: Center slice index
            volume_shape: (Z, Y, X) shape of CT volume
            margin_xy: Margin to add in X/Y dimensions
            margin_z: Margin to add in Z dimension
            
        Returns:
            ((z_start, z_end), (y_start, y_end), (x_start, x_end)) crop bounds
        """
        z_dim, y_dim, x_dim = volume_shape
        xmin, ymin, xmax, ymax = bbox
        
        # Compute bounds with margins
        x_start = max(0, xmin - margin_xy)
        x_end = min(x_dim, xmax + margin_xy)
        y_start = max(0, ymin - margin_xy)
        y_end = min(y_dim, ymax + margin_xy)
        z_start = max(0, key_slice_idx - margin_z)
        z_end = min(z_dim, key_slice_idx + margin_z + 1)
        
        return (z_start, z_end), (y_start, y_end), (x_start, x_end)
    
    def segment_lesion_from_bbox(
        self,
        ct_volume: np.ndarray,
        key_slice_idx: int,
        bbox: Tuple[int, int, int, int],
        window_lower: float = -1000,
        window_upper: float = 40,
        bidirectional: bool = True,
        keep_largest_component: bool = True,
        use_crop: bool = True,
        crop_margin_xy: int = 32,
        crop_margin_z: int = 15,
    ) -> np.ndarray:
        """
        Segment lesion in 3D CT volume using bbox prompt on one slice.
        
        This method:
        1. Optionally crops a subvolume around the bbox for performance
        2. Preprocesses the volume (windowing + normalization)
        3. Resizes to 512x512 for MedSAM2
        4. Runs inference and propagates through slices
        5. Resizes masks back to original resolution
        6. Pastes result into full volume coordinates
        
        Args:
            ct_volume: 3D CT volume in HU, shape (Z, Y, X)
            key_slice_idx: Slice index where the bbox annotation is
            bbox: Bounding box as (xmin, ymin, xmax, ymax) in original image coords
            window_lower: CT window lower bound (HU)
            window_upper: CT window upper bound (HU)
            bidirectional: If True, propagate both forward and backward
            keep_largest_component: If True, keep only largest connected component
            use_crop: If True, crop around bbox for faster inference
            crop_margin_xy: Margin around bbox in X/Y for cropping
            crop_margin_z: Margin around key slice in Z for cropping
            
        Returns:
            Binary mask of shape (Z, Y, X) with same dimensions as input
        """
        z_dim, y_dim, x_dim = ct_volume.shape
        
        # Validate key_slice_idx
        if key_slice_idx < 0 or key_slice_idx >= z_dim:
            raise ValueError(f"key_slice_idx {key_slice_idx} out of range [0, {z_dim})")
        
        # Initialize full output mask
        full_seg = np.zeros((z_dim, y_dim, x_dim), dtype=np.uint8)
        
        if use_crop:
            # Crop-based inference for performance
            (z_start, z_end), (y_start, y_end), (x_start, x_end) = self._compute_crop_bounds(
                bbox, key_slice_idx, (z_dim, y_dim, x_dim),
                margin_xy=crop_margin_xy, margin_z=crop_margin_z
            )
            
            # Extract crop
            crop = ct_volume[z_start:z_end, y_start:y_end, x_start:x_end]
            crop_z, crop_y, crop_x = crop.shape
            
            # Adjust key_slice_idx and bbox to crop coordinates
            crop_key_slice = key_slice_idx - z_start
            xmin, ymin, xmax, ymax = bbox
            crop_bbox = (xmin - x_start, ymin - y_start, xmax - x_start, ymax - y_start)
            
            # Run inference on crop
            crop_seg = self._segment_volume(
                crop, crop_key_slice, crop_bbox,
                window_lower, window_upper, bidirectional
            )
            
            # Apply LCC on cropped result if requested
            if keep_largest_component and np.max(crop_seg) > 0:
                crop_seg = self._get_largest_connected_component(crop_seg)
            
            # Paste back into full mask
            full_seg[z_start:z_end, y_start:y_end, x_start:x_end] = crop_seg
        else:
            # Full volume inference (slower but may be more accurate for large lesions)
            full_seg = self._segment_volume(
                ct_volume, key_slice_idx, bbox,
                window_lower, window_upper, bidirectional
            )
            
            if keep_largest_component and np.max(full_seg) > 0:
                full_seg = self._get_largest_connected_component(full_seg)
        
        return full_seg
    
    def _segment_volume(
        self,
        volume: np.ndarray,
        key_slice_idx: int,
        bbox: Tuple[int, int, int, int],
        window_lower: float,
        window_upper: float,
        bidirectional: bool,
    ) -> np.ndarray:
        """
        Core segmentation logic on a (possibly cropped) volume.
        
        Handles preprocessing, resizing, inference, and mask resizing.
        """
        z_dim, y_dim, x_dim = volume.shape
        
        # Preprocess volume
        preprocessed = self._preprocess_ct_volume(volume, window_lower, window_upper)
        
        # Resize to RGB for MedSAM2 (512x512)
        img_tensor = self._resize_to_rgb(preprocessed, target_size=MODEL_SIZE)
        
        # Scale bbox to 512x512 space
        scale_x = MODEL_SIZE / x_dim
        scale_y = MODEL_SIZE / y_dim
        xmin, ymin, xmax, ymax = bbox
        scaled_bbox = np.array([
            xmin * scale_x,
            ymin * scale_y,
            xmax * scale_x,
            ymax * scale_y,
        ])
        
        # Initialize output mask at original resolution
        seg_3d = np.zeros((z_dim, y_dim, x_dim), dtype=np.uint8)
        
        with torch.inference_mode():
            with _autocast_context(self.device, dtype=torch.float16):
                # Initialize predictor state
                # Pass original dimensions so MedSAM2 knows output size
                inference_state = self.predictor.init_state(img_tensor, MODEL_SIZE, MODEL_SIZE)
                
                # Add bbox prompt on key slice
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=key_slice_idx,
                    obj_id=1,
                    box=scaled_bbox,
                )
                
                # Propagate forward
                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                    # Mask is at model resolution (512x512)
                    mask_512 = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
                    # Resize to original resolution
                    mask_orig = _resize_mask_to_original(mask_512, y_dim, x_dim)
                    # Assign using proper indexing
                    seg_3d[out_frame_idx] = np.maximum(seg_3d[out_frame_idx], mask_orig)
                
                if bidirectional:
                    # Reset and propagate backward
                    self.predictor.reset_state(inference_state)
                    
                    # Re-add prompt
                    _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=key_slice_idx,
                        obj_id=1,
                        box=scaled_bbox,
                    )
                    
                    # Propagate backward
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                        inference_state, reverse=True
                    ):
                        mask_512 = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
                        mask_orig = _resize_mask_to_original(mask_512, y_dim, x_dim)
                        seg_3d[out_frame_idx] = np.maximum(seg_3d[out_frame_idx], mask_orig)
                
                self.predictor.reset_state(inference_state)
        
        return seg_3d
    
    def segment_multiple_bboxes(
        self,
        ct_volume: np.ndarray,
        bboxes_by_slice: Dict[int, List[Tuple[int, int, int, int]]],
        window_lower: float = -1000,
        window_upper: float = 400,
        min_component_voxels: int = 50,
        use_crop: bool = True,
    ) -> np.ndarray:
        """
        Segment multiple lesions from bboxes on different slices.
        
        Uses crop-based inference for performance - each bbox is processed
        on a cropped subvolume rather than the full CT.
        
        Multi-lesion preservation: Instead of global LCC, applies per-lesion
        LCC and then removes only tiny noise components from the final union.
        
        Args:
            ct_volume: 3D CT volume in HU, shape (Z, Y, X)
            bboxes_by_slice: Dict mapping slice_idx -> list of bboxes
            window_lower: CT window lower bound (HU)
            window_upper: CT window upper bound (HU)
            min_component_voxels: Remove components smaller than this (default 50)
            use_crop: Use crop-based inference for speed
            
        Returns:
            Combined binary mask of all segmented lesions
        """
        z_dim, y_dim, x_dim = ct_volume.shape
        combined_mask = np.zeros((z_dim, y_dim, x_dim), dtype=np.uint8)
        
        total_bboxes = sum(len(bboxes) for bboxes in bboxes_by_slice.values())
        processed = 0
        
        for slice_idx, bboxes in bboxes_by_slice.items():
            for bbox in bboxes:
                processed += 1
                try:
                    # Use crop-based inference with per-lesion LCC
                    lesion_mask = self.segment_lesion_from_bbox(
                        ct_volume=ct_volume,
                        key_slice_idx=slice_idx,
                        bbox=bbox,
                        window_lower=window_lower,
                        window_upper=window_upper,
                        keep_largest_component=True,  # LCC per lesion
                        use_crop=use_crop,
                    )
                    # Union with existing mask (preserves multiple lesions)
                    combined_mask = np.maximum(combined_mask, lesion_mask)
                    
                    if processed % 5 == 0 or processed == total_bboxes:
                        print(f"Processed {processed}/{total_bboxes} bounding boxes")
                        
                except Exception as e:
                    print(f"Warning: Failed to segment bbox {bbox} on slice {slice_idx}: {e}")
                    continue
        
        # Remove only tiny noise components, preserve all significant lesions
        if min_component_voxels > 0 and np.max(combined_mask) > 0:
            combined_mask = _remove_small_components(combined_mask, min_voxels=min_component_voxels)
        
        return combined_mask
