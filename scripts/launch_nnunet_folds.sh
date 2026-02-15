#!/usr/bin/env bash
# Launch nnU-Net training for Dataset503_LIDC_SUBSET across all 4 GPUs.
# Each fold runs in its own tmux window inside session "nnunet".
#
# Fold 0: GPU 0 (resume from epoch 21)
# Fold 1: GPU 1 (fresh)
# Fold 2: GPU 2 (fresh)
# Fold 3: GPU 3 (fresh)
#
# Usage:
#   bash scripts/launch_nnunet_folds.sh          # launch all
#   bash scripts/launch_nnunet_folds.sh 0        # launch fold 0 only
#   bash scripts/launch_nnunet_folds.sh 1 2 3    # launch folds 1,2,3

set -euo pipefail
cd /home/asafz/projects/radiomics10/RADIOMICS10

SESSION="nnunet"
VENV="/home/asafz/projects/radiomics10/RADIOMICS10/venv/bin/activate"
RAW="/mnt/metlab27/D/Input/Radiomics/T_IL Radiomics10/LIDC-IDRI/Output/nnunet/nnUNet_raw"
PREP="/home/asafz/projects/radiomics10/nnUNet_preprocessed_local"
RES="/home/asafz/projects/radiomics10/nnUNet_results_local"
DATASET=503
LOGDIR="/home/asafz/projects/radiomics10/nnUNet_results_local/logs"

mkdir -p "$LOGDIR"

# Which folds to launch (default: all 4)
if [ $# -eq 0 ]; then
    FOLDS=(0 1 2 3)
else
    FOLDS=("$@")
fi

# Create tmux session if it doesn't exist
tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION" -n "monitor"

for FOLD in "${FOLDS[@]}"; do
    GPU=$FOLD
    LOGFILE="$LOGDIR/fold${FOLD}_$(date +%Y%m%d_%H%M%S).log"

    # Fold 0 resumes; others start fresh
    if [ "$FOLD" -eq 0 ]; then
        CONTINUE="--c"
    else
        CONTINUE=""
    fi

    WINDOW="fold${FOLD}"
    CMD="source $VENV && \
export nnUNet_raw='$RAW' && \
export nnUNet_preprocessed='$PREP' && \
export nnUNet_results='$RES' && \
export CUDA_VISIBLE_DEVICES=$GPU && \
echo '=== Fold $FOLD on GPU $GPU  started $(date) ===' | tee $LOGFILE && \
nnUNetv2_train $DATASET 3d_fullres $FOLD --npz $CONTINUE 2>&1 | tee -a $LOGFILE ; \
echo '=== Fold $FOLD finished $(date) ===' | tee -a $LOGFILE"

    # Create window and send command
    tmux new-window -t "$SESSION" -n "$WINDOW" 2>/dev/null || true
    tmux send-keys -t "$SESSION:$WINDOW" "$CMD" C-m

    echo "Launched fold $FOLD on GPU $GPU -> tmux window '$SESSION:$WINDOW'"
    echo "  Log: $LOGFILE"

    # Small delay to avoid race conditions on shared filesystem
    sleep 2
done

echo ""
echo "All folds launched. Monitor with:"
echo "  tmux attach -t $SESSION"
echo "  tmux select-window -t $SESSION:fold0   # switch to fold 0"
echo "  tmux select-window -t $SESSION:fold1   # switch to fold 1"
echo ""
echo "Logs: ls -lh $LOGDIR/"
