#!/bin/bash
#
# =============================================================================
# All Dataset Training Script - train_all_parallel.sh
# Description: Train all datasets on GPU 0-6 in parallel
# Requirements:
#   - Run training tasks in parallel on 7 GPUs (0-6)
#   - Test all datasets with 2, 16, 64 shot
#   - Test sdd, dagm, dtd with full shot
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
RESULT_DIR="./all_results"

# Log file
LOG_FILE="${PROJECT_DIR}/all_training.log"

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
            --batch_size 16 \
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
echo "All Dataset Training Script Started"
echo "Project: $PROJECT_DIR"
echo "Log file: $LOG_FILE"
echo "GPU allocation: GPU 0-6 (7 GPUs in parallel)"
echo "Mode: Parallel (7 GPUs simultaneously)"
echo "=========================================="
echo ""

# Create result directory
mkdir -p "$RESULT_DIR"

# Dataset configuration: "dataset_name|dataset_path"
declare -A DATASET_CONFIG

# All datasets
DATASET_CONFIG["btech"]="btech|${BASE_DATA_DIR}/BTech_Dataset_transformed"
DATASET_CONFIG["cvc300"]="cvc300|${BASE_DATA_DIR}/CVC-300"
DATASET_CONFIG["cvc_clinicdb"]="cvc_clinicdb|${BASE_DATA_DIR}/CVC-ClinicDB"
DATASET_CONFIG["cvc_colondb"]="cvc_colondb|${BASE_DATA_DIR}/CVC-ColonDB"
DATASET_CONFIG["dagm"]="dagm|${BASE_DATA_DIR}/DAGM/DAGM_KaggleUpload"
DATASET_CONFIG["dtd"]="dtd|${BASE_DATA_DIR}/DTD-Synthetic/DTD-Synthetic"
DATASET_CONFIG["kvasir"]="kvasir|${BASE_DATA_DIR}/Kvasir"
DATASET_CONFIG["medad"]="medad|${BASE_DATA_DIR}/MedAD"
DATASET_CONFIG["mpdd"]="mpdd|${BASE_DATA_DIR}/MPDD"
DATASET_CONFIG["mvtec"]="mvtec|${BASE_DATA_DIR}/mvtec"
DATASET_CONFIG["sdd"]="sdd|${BASE_DATA_DIR}/SDD"
DATASET_CONFIG["visa"]="visa|${BASE_DATA_DIR}/visa"

# K-shot values for few-shot (2, 16, 64)
K_SHOTS=(2 16 64)

# GPU list (0-6) - All available GPUs
GPUS=(0 1 2 3 4 5 6)

# Generate all tasks
# Task format: "dataset_name|dataset_path|k_shot|save_path"
declare -a ALL_TASKS=()

# Add few-shot tasks for all datasets (2, 16, 64 shot)
for dataset_name in "${!DATASET_CONFIG[@]}"; do
    IFS="|" read -r DATASET_NAME DATASET_PATH <<< "${DATASET_CONFIG[$dataset_name]}"
    for k_shot in "${K_SHOTS[@]}"; do
        FOLDER_NAME=$(basename "$DATASET_PATH")
        SAVE_PATH="${RESULT_DIR}/${FOLDER_NAME}/shot-${k_shot}"
        task="$DATASET_NAME|$DATASET_PATH|$k_shot|$SAVE_PATH"
        ALL_TASKS+=("$task")
    done
done

# Add full-shot tasks for sdd, dagm, dtd (k_shot=0)
for dataset_name in "sdd" "dagm" "dtd"; do
    IFS="|" read -r DATASET_NAME DATASET_PATH <<< "${DATASET_CONFIG[$dataset_name]}"
    FOLDER_NAME=$(basename "$DATASET_PATH")
    SAVE_PATH="${RESULT_DIR}/${FOLDER_NAME}/shot-full"
    task="$DATASET_NAME|$DATASET_PATH|0|$SAVE_PATH"
    ALL_TASKS+=("$task")
done

echo "Total tasks: ${#ALL_TASKS[@]}"
echo "Tasks per GPU (round-robin):"
echo ""

# Distribute tasks to GPUs (round-robin)
# GPU_TASKS[i] = list of tasks for GPU i (GPU 0-6)
declare -a GPU_TASKS_0=()  # GPU 0
declare -a GPU_TASKS_1=()  # GPU 1
declare -a GPU_TASKS_2=()  # GPU 2
declare -a GPU_TASKS_3=()  # GPU 3
declare -a GPU_TASKS_4=()  # GPU 4
declare -a GPU_TASKS_5=()  # GPU 5
declare -a GPU_TASKS_6=()  # GPU 6

for i in "${!ALL_TASKS[@]}"; do
    task="${ALL_TASKS[$i]}"
    gpu_idx=$(( i % 7 ))
    
    case $gpu_idx in
        0) GPU_TASKS_0+=("$task") ;;  # GPU 0
        1) GPU_TASKS_1+=("$task") ;;  # GPU 1
        2) GPU_TASKS_2+=("$task") ;;  # GPU 2
        3) GPU_TASKS_3+=("$task") ;;  # GPU 3
        4) GPU_TASKS_4+=("$task") ;;  # GPU 4
        5) GPU_TASKS_5+=("$task") ;;  # GPU 5
        6) GPU_TASKS_6+=("$task") ;;  # GPU 6
    esac
done

# Print task distribution
echo "GPU 0 tasks: ${#GPU_TASKS_0[@]}"
echo "GPU 1 tasks: ${#GPU_TASKS_1[@]}"
echo "GPU 2 tasks: ${#GPU_TASKS_2[@]}"
echo "GPU 3 tasks: ${#GPU_TASKS_3[@]}"
echo "GPU 4 tasks: ${#GPU_TASKS_4[@]}"
echo "GPU 5 tasks: ${#GPU_TASKS_5[@]}"
echo "GPU 6 tasks: ${#GPU_TASKS_6[@]}"
echo ""

# Launch GPU workers in parallel
echo "Launching GPU workers..."
echo ""

# Worker for GPU 0
run_gpu_tasks 0 "${GPU_TASKS_0[@]}" &
PIDS+=($!)

# Worker for GPU 1
run_gpu_tasks 1 "${GPU_TASKS_1[@]}" &
PIDS+=($!)

# Worker for GPU 2
run_gpu_tasks 2 "${GPU_TASKS_2[@]}" &
PIDS+=($!)

# Worker for GPU 3
run_gpu_tasks 3 "${GPU_TASKS_3[@]}" &
PIDS+=($!)

# Worker for GPU 4
run_gpu_tasks 4 "${GPU_TASKS_4[@]}" &
PIDS+=($!)

# Worker for GPU 5
run_gpu_tasks 5 "${GPU_TASKS_5[@]}" &
PIDS+=($!)

# Worker for GPU 6
run_gpu_tasks 6 "${GPU_TASKS_6[@]}" &
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
echo "All Training Tasks Completed Successfully!"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

exit 0
