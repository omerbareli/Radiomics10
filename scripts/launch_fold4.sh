#!/usr/bin/env bash
# Launch nnU-Net training for fold 4 (the 5th fold) on GPU 0.
#
# Key differences from launch_nnunet_folds.sh:
#   - Hardcoded GPU=0, FOLD=4
#   - No --npz flag (saves ~20 GB of validation probability maps)
#   - Fresh start (no --c)
#
# Usage:
#   bash scripts/launch_fold4.sh

set -euo pipefail
cd /home/asafz/projects/radiomics10/RADIOMICS10

SESSION="nnunet"
VENV="/home/asafz/projects/radiomics10/RADIOMICS10/venv/bin/activate"
RAW="/mnt/metlab27/D/Input/Radiomics/T_IL Radiomics10/LIDC-IDRI/Output/nnunet/nnUNet_raw"
PREP="/home/asafz/projects/radiomics10/nnUNet_preprocessed_local"
RES="/home/asafz/projects/radiomics10/nnUNet_results_local"
DATASET=503
LOGDIR="$RES/logs"
GPU=0
FOLD=4

mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/fold${FOLD}_$(date +%Y%m%d_%H%M%S).log"

# Create tmux session if it doesn't exist
tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION" -n "monitor"

WINDOW="fold${FOLD}"
CMD="source $VENV && \
export nnUNet_raw='$RAW' && \
export nnUNet_preprocessed='$PREP' && \
export nnUNet_results='$RES' && \
export CUDA_VISIBLE_DEVICES=$GPU && \
echo '=== Fold $FOLD on GPU $GPU  started \$(date) ===' | tee $LOGFILE && \
nnUNetv2_train $DATASET 3d_fullres $FOLD 2>&1 | tee -a $LOGFILE ; \
echo '=== Fold $FOLD finished \$(date) ===' | tee -a $LOGFILE"

# Create window and send command
tmux new-window -t "$SESSION" -n "$WINDOW" 2>/dev/null || true
tmux send-keys -t "$SESSION:$WINDOW" "$CMD" C-m

echo "Launched fold $FOLD on GPU $GPU -> tmux window '$SESSION:$WINDOW'"
echo "  Log: $LOGFILE"
echo ""
echo "Monitor with:"
echo "  tmux attach -t $SESSION"
echo "  tmux select-window -t $SESSION:$WINDOW"
echo "  tail -f $LOGFILE"
