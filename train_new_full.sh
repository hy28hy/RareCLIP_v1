#!/bin/bash
#
# =============================================================================
# Full Training Script - train_new_full.sh
# Description: Train full model on GPU 0,1,2 in parallel
# Requirements:
#   - Run training tasks in parallel on multiple GPUs
#   - Support Ctrl+C interruption
#   - Kill all child processes on interruption
# =============================================================================

# Enable exit on error and trace
set -e
set -o pipefail

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate RareCLIP

# Project directory
PROJECT_DIR="/data/chenxuwu/zihaowan_workplace/RareCLIP"
cd "$PROJECT_DIR" || exit 1

# Base data directory
BASE_DATA_DIR="/data/chenxuwu/zihaowan_workplace/dataset"

# Log file
LOG_FILE="${PROJECT_DIR}/full.log"

# Array to store background PIDs
declare -a PIDS=()

# Cleanup function - called on Ctrl+C
cleanup() {
    echo ""
    echo "=========================================="
    echo "Received Ctrl+C (SIGINT)"
    echo "Killing all training processes..."
    echo "=========================================="

    # Kill all background processes
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing process $pid"
            kill -9 "$pid" 2>/dev/null || true
            kill -9 -"$pid" 2>/dev/null || true
        fi
    done

    # Kill all child processes of this script
    echo "Killing all child processes..."
    pkill -9 -P $$ 2>/dev/null || true

    echo ""
    echo "All processes killed. Exiting."
    exit 130  # Standard exit code for Ctrl+C
}

# Set trap for Ctrl+C (SIGINT) and SIGTERM
trap cleanup SIGINT SIGTERM

# Function to run training in background
run_training() {
    local gpu_id=$1
    local dataset_name=$2
    local dataset_path=$3
    local save_path=$4

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting training: GPU=$gpu_id, Dataset=$dataset_name"

    # Run training in background, redirect output to log file
    python train.py \
        --train "$dataset_name" \
        --train_set_path "$dataset_path" \
        --gpu "$gpu_id" \
        --epoch 5 \
        --batch_size 8 \
        --save_path "$save_path" \
        --k_shot 0 >> "$LOG_FILE" 2>&1 &

    # Store PID
    PIDS+=($!)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launched training on GPU $gpu_id (PID: ${PIDS[-1]})"
}

# =============================================================================
# Main Execution
# =============================================================================

echo ""
echo "=========================================="
echo "Full Training Script Started"
echo "Project: $PROJECT_DIR"
echo "Log file: $LOG_FILE"
echo "GPU allocation: GPU 0 → dagm, GPU 1 → dtd, GPU 2 → sdd"
echo "Mode: Parallel (all GPUs simultaneously)"
echo "=========================================="
echo ""

# Dataset configuration: "dataset_name|dataset_path|save_path"
# GPU 0 → dagm, GPU 1 → dtd, GPU 2 → sdd
declare -a DATASETS
DATASETS[0]="dagm|${BASE_DATA_DIR}/DAGM/DAGM_KaggleUpload|./exps/DAGM_KaggleUpload_train"
DATASETS[1]="dtd|${BASE_DATA_DIR}/DTD-Synthetic/DTD-Synthetic|./exps/DTD-Synthetic_train"
DATASETS[2]="sdd|${BASE_DATA_DIR}/SDD|./exps/SDD_train"

# GPU list
GPUS=(0 1 2)

# Run training in parallel (all at once)
for i in "${!GPUS[@]}"; do
    IFS="|" read -r DATASET_NAME DATASET_PATH SAVE_PATH <<< "${DATASETS[$i]}"
    run_training "${GPUS[$i]}" "$DATASET_NAME" "$DATASET_PATH" "$SAVE_PATH"
done

echo ""
echo "All training tasks launched. Waiting for completion..."
echo "PIDs: ${PIDS[*]}"
echo ""

# Wait for all background processes to complete
FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid"
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training (PID $pid) FAILED with exit code $exit_code"
        FAILED=1
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training (PID $pid) completed successfully"
    fi
done

if [ $FAILED -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "Some training tasks FAILED"
    echo "=========================================="
    exit 1
fi

echo ""
echo "=========================================="
echo "All Full Training Tasks Completed Successfully!"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

exit 0
