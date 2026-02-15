# nnU-Net Permission Patch

## Problem
Network mounts (NTFS/SMB/CIFS) often do not support Linux `chmod` operations.
nnU-Net v2.1's `default_experiment_planner.py` uses `shutil.copy()` which 
internally calls `copymode()` to preserve file permissions. This fails with:
```
PermissionError: [Errno 1] Operation not permitted
```

## Workaround
This patch changes `shutil.copy()` to `shutil.copyfile()` which copies only
the file content without attempting to preserve permissions.

## How to Apply
```bash
cd $VIRTUAL_ENV/lib/python3.11/site-packages/nnunetv2/
patch -p1 < /path/to/nnunet_copy_permission.patch
```

## How to Revert
```bash
cd $VIRTUAL_ENV/lib/python3.11/site-packages/nnunetv2/
patch -R -p1 < /path/to/nnunet_copy_permission.patch
```

## Preferred Long-Term Fix
Use a local Linux filesystem for `nnUNet_preprocessed` and `nnUNet_results`,
then copy final artifacts to the network mount after training completes.
This avoids all NTFS/SMB permission issues.

## File Modified
- `nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py`
  - Line 461: `shutil.copy` → `shutil.copyfile`

## Tested With
- nnU-Net v2.1
- Python 3.11
