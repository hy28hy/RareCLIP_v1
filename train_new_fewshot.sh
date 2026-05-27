#!/bin/bash
#
# =============================================================================
# Few-shot Training Script - train_new_fewshot.sh
# Description: Train few-shot model on GPU 3,4,5,6,7 in parallel
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

# Result directory
RESULT_DIR="./fewshot_results"

# Log file
LOG_FILE="${PROJECT_DIR}/fewshot.log"

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

# Function to run all tasks for a GPU (sequentially)
run_gpu_tasks() {
    local gpu_id=$1
    shift
    local tasks=("$@")

    for task in "${tasks[@]}"; do
        IFS="|" read -r DATASET_NAME DATASET_PATH K_SHOT SAVE_PATH <<< "$task"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu_id: Starting $DATASET_NAME (k=$K_SHOT)"

        # Run python in subshell to capture exit code with set -e
        (python train.py \
            --train "$DATASET_NAME" \
            --train_set_path "$DATASET_PATH" \
            --gpu "$gpu_id" \
            --epoch 5 \
            --batch_size 8 \
            --k_shot "$K_SHOT" \
            --save_path "$SAVE_PATH" >> "$LOG_FILE" 2>&1)
        # Note: the subshell inherits set -e, so if python fails, subshell exits with python's exit code

        exit_code=$?

        if [ $exit_code -ne 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu_id: FAILED $DATASET_NAME (k=$K_SHOT) with exit code $exit_code"
            return $exit_code
        fi

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu_id: Completed $DATASET_NAME (k=$K_SHOT)"
    done

    return 0
}

# =============================================================================
# Main Execution
# =============================================================================

echo ""
echo "=========================================="
echo "Few-shot Training Script Started"
echo "Project: $PROJECT_DIR"
echo "Log file: $LOG_FILE"
echo "GPU allocation: GPU 3,4,5,6,7 (round-robin for 9 tasks)"
echo "Mode: Parallel (5 GPUs simultaneously)"
echo "=========================================="
echo ""

# Create result directory
mkdir -p "$RESULT_DIR"

# Dataset configuration: "dataset_name|dataset_path"
declare -A DATASET_CONFIG
DATASET_CONFIG["dagm"]="dagm|${BASE_DATA_DIR}/DAGM/DAGM_KaggleUpload"
DATASET_CONFIG["dtd"]="dtd|${BASE_DATA_DIR}/DTD-Synthetic/DTD-Synthetic"
DATASET_CONFIG["sdd"]="sdd|${BASE_DATA_DIR}/SDD"

# K-shot values
K_SHOTS=(2 16 64)

# GPU list
GPUS=(3 4 5 6 7)

# Generate all 9 tasks and assign to GPUs (round-robin)
# gpu_tasks is an array of arrays: GPU_TASKS[i] = list of tasks for GPU i
declare -a GPU_TASKS_0=()
declare -a GPU_TASKS_1=()
declare -a GPU_TASKS_2=()
declare -a GPU_TASKS_3=()
declare -a GPU_TASKS_4=()

gpu_idx=0
for dataset_name in "${!DATASET_CONFIG[@]}"; do
    IFS="|" read -r DATASET_NAME DATASET_PATH <<< "${DATASET_CONFIG[$dataset_name]}"
    for k_shot in "${K_SHOTS[@]}"; do
        FOLDER_NAME=$(basename "$DATASET_PATH")
        SAVE_PATH="${RESULT_DIR}/${FOLDER_NAME}/shot-${k_shot}"
        task="$DATASET_NAME|$DATASET_PATH|$k_shot|$SAVE_PATH"

        # Assign to GPU (round-robin)
        case $gpu_idx in
            0) GPU_TASKS_0+=("$task") ;;
            1) GPU_TASKS_1+=("$task") ;;
            2) GPU_TASKS_2+=("$task") ;;
            3) GPU_TASKS_3+=("$task") ;;
            4) GPU_TASKS_4+=("$task") ;;
        esac

        gpu_idx=$(( (gpu_idx + 1) % 5 ))
    done
done

# Launch GPU workers in parallel
echo "Launching GPU workers..."
echo "GPU 3 tasks: ${#GPU_TASKS_0[@]}"
echo "GPU 4 tasks: ${#GPU_TASKS_1[@]}"
echo "GPU 5 tasks: ${#GPU_TASKS_2[@]}"
echo "GPU 6 tasks: ${#GPU_TASKS_3[@]}"
echo "GPU 7 tasks: ${#GPU_TASKS_4[@]}"
echo ""

# Worker for GPU 3 (index 0)
run_gpu_tasks 3 "${GPU_TASKS_0[@]}" &
PIDS+=($!)

# Worker for GPU 4 (index 1)
run_gpu_tasks 4 "${GPU_TASKS_1[@]}" &
PIDS+=($!)

# Worker for GPU 5 (index 2)
run_gpu_tasks 5 "${GPU_TASKS_2[@]}" &
PIDS+=($!)

# Worker for GPU 6 (index 3)
run_gpu_tasks 6 "${GPU_TASKS_3[@]}" &
PIDS+=($!)

# Worker for GPU 7 (index 4)
run_gpu_tasks 7 "${GPU_TASKS_4[@]}" &
PIDS+=($!)

echo "All GPU workers launched. PIDs: ${PIDS[*]}"
echo "Waiting for all workers to complete..."
echo ""

# Wait for all and check exit codes
FAILED=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    wait "$pid"
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU worker $i (PID $pid) FAILED with exit code $exit_code"
        FAILED=1
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU worker $i (PID $pid) completed successfully"
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
echo "All Few-shot Training Tasks Completed Successfully!"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

exit 0
