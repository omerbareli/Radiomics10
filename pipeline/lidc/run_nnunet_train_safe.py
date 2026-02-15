# pipeline/lidc/run_nnunet_train_safe.py
"""Run nnUNet training with cuDNN disabled for Pascal GPU compatibility.

Monkey-patches torch.backends.cudnn to prevent nnUNet from enabling
cudnn.benchmark, which causes CUDNN_STATUS_EXECUTION_FAILED on TITAN Xp.

Usage:
    python -m pipeline.lidc.run_nnunet_train_safe 503 3d_fullres 0 --npz
"""
import torch.backends.cudnn as _cudnn_mod

# Completely prevent cudnn benchmark from being enabled.
# nnUNet's run_training() sets cudnn.benchmark = True internally,
# so we need to make the setter a no-op.
_cudnn_mod.enabled = False  # Disable cuDNN entirely

# Make benchmark setter a no-op so nnUNet can't re-enable it
_original_benchmark = _cudnn_mod.benchmark


class _CudnnBenchmarkBlocker:
    """Descriptor that blocks setting benchmark to True."""
    def __get__(self, obj, objtype=None):
        return False
    def __set__(self, obj, value):
        pass  # no-op: prevent nnUNet from setting benchmark = True


# We can't easily use descriptors on the module, so instead
# monkey-patch the torch.backends.cudnn module's __setattr__
_orig_setattr = type(_cudnn_mod).__setattr__

def _blocked_setattr(self, name, value):
    if name == 'benchmark':
        return  # block
    if name == 'enabled':
        return  # block - keep disabled
    _orig_setattr(self, name, value)

type(_cudnn_mod).__setattr__ = _blocked_setattr

# Now import and run nnunet
from nnunetv2.run.run_training import run_training_entry

if __name__ == "__main__":
    run_training_entry()
