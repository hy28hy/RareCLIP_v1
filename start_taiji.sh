#!/bin/bash
#
# TaiJi startup script for RareCLIP training
# - Install dependencies
# - Download dataset from ModelScope
# - Run 8-GPU parallel training
#

set -e

echo "=========================================="
echo "RareCLIP TaiJi Startup Script"
echo "Start time: $(date)"
echo "=========================================="

# =============================================================================
# Step 0: Get RareCLIP code
# =============================================================================
echo ""
echo "===== Step 0: Getting RareCLIP code ====="

CODE_DIR="$PWD"

# If running from /jizhi/jizhi2/worker/trainer, clone repo
if [[ "$CODE_DIR" == "/jizhi/jizhi2/worker/trainer" ]] || [[ ! -f "$CODE_DIR/train.py" ]]; then
    echo "Not in RareCLIP directory, cloning from git..."
    cd /workspace
    if [ -d RareCLIP_v1 ]; then
        rm -rf RareCLIP_v1
    fi
    git clone https://github.com/hy28hy/RareCLIP_v1.git
    cd RareCLIP_v1
    CODE_DIR="$PWD"
else
    echo "Already in RareCLIP directory: $CODE_DIR"
fi

echo "Working directory: $CODE_DIR"
ls -la *.py 2>/dev/null | head -10

if [ ! -f "train.py" ]; then
    echo "ERROR: train.py not found in $CODE_DIR"
    exit 1
fi

echo "===== Code ready! ====="

# =============================================================================
# Step 1: Install dependencies
# =============================================================================
echo ""
echo "===== Step 1: Installing dependencies ====="

pip install --index-url https://mirrors.tencent.com/pypi/simple/ \
    einops ftfy opencv-python pandas Pillow regex \
    scikit-image scikit-learn tabulate tqdm timm modelscope

echo "===== Dependencies installed! ====="

# =============================================================================
# Step 2: Download dataset from ModelScope
# =============================================================================
echo ""
echo "===== Step 2: Downloading dataset from ModelScope ====="

# Create download_dataset.py on the fly
cat > download_dataset.py << 'DATASET_EOF'
#!/usr/bin/env python3
"""Download RareCLIP datasets from ModelScope"""
import os
from modelscope import snapshot_download

# Dataset mapping: local_name -> modelscope_dataset_id
DATASETS = {
    "BTech_Dataset_transformed": "zihaowan/BTech_Dataset_transformed",
    "CVC-300": "zihaowan/CVC-300",
    "CVC-ClinicDB": "zihaowan/CVC-ClinicDB",
    "CVC-ColonDB": "zihaowan/CVC-ColonDB",
    "DAGM": "zihaowan/DAGM",
    "DTD-Synthetic": "zihaowan/DTD-Synthetic",
    "Kvasir": "zihaowan/Kvasir",
    "MedAD": "zihaowan/MedAD",
    "MPDD": "zihaowan/MPDD",
    "mvtec": "zihaowan/mvtec",
    "SDD": "zihaowan/SDD",
    "visa": "zihaowan/visa",
}

def main():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    os.makedirs(base_dir, exist_ok=True)
    
    for name, dataset_id in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Downloading: {name}")
        print(f"Dataset ID: {dataset_id}")
        print(f"{'='*60}")
        try:
            local_path = snapshot_download(dataset_id, cache_dir=base_dir)
            print(f"Downloaded to: {local_path}")
        except Exception as e:
            print(f"ERROR downloading {name}: {e}")
            continue
    
    print("\nAll datasets downloaded!")

if __name__ == "__main__":
    main()
DATASET_EOF

# Ensure modelscope is installed for the Python that will run download_dataset.py
# Find the correct Python path
PYTHON_BIN=$(which python3)
echo "Python binary: $PYTHON_BIN"
$PYTHON_BIN -m pip install --index-url https://mirrors.tencent.com/pypi/simple/ modelscope

# Run download_dataset.py with the same Python
$PYTHON_BIN download_dataset.py

if [ $? -ne 0 ]; then
    echo "ERROR: Dataset download failed"
    exit 1
fi

echo "===== Dataset ready! ====="

# =============================================================================
# Step 3: Run 8-GPU parallel training
# =============================================================================
echo ""
echo "===== Step 3: Starting 8-GPU parallel training ====="
echo "Start time: $(date)"

# Create training script
cat > train_all_parallel_taiji.sh << 'TRAIN_EOF'
#!/bin/bash
set -e

# Project directory
PROJECT_DIR="$PWD"
DATA_BASE_DIR="${PROJECT_DIR}/datasets"
RESULT_DIR="./all_results"
LOG_FILE="${PROJECT_DIR}/all_training.log"

mkdir -p "$RESULT_DIR"

# Dataset configuration
declare -A DATASET_CONFIG
DATASET_CONFIG["btech"]="btech|${DATA_BASE_DIR}/BTech_Dataset_transformed"
DATASET_CONFIG["cvc300"]="cvc300|${DATA_BASE_DIR}/CVC-300"
DATASET_CONFIG["cvc_clinicdb"]="cvc_clinicdb|${DATA_BASE_DIR}/CVC-ClinicDB"
DATASET_CONFIG["cvc_colondb"]="cvc_colondb|${DATA_BASE_DIR}/CVC-ColonDB"
DATASET_CONFIG["dagm"]="dagm|${DATA_BASE_DIR}/DAGM/DAGM_KaggleUpload"
DATASET_CONFIG["dtd"]="dtd|${DATA_BASE_DIR}/DTD-Synthetic/DTD-Synthetic"
DATASET_CONFIG["kvasir"]="kvasir|${DATA_BASE_DIR}/Kvasir"
DATASET_CONFIG["medad"]="medad|${DATA_BASE_DIR}/MedAD"
DATASET_CONFIG["mpdd"]="mpdd|${DATA_BASE_DIR}/MPDD"
DATASET_CONFIG["mvtec"]="mvtec|${DATA_BASE_DIR}/mvtec"
DATASET_CONFIG["sdd"]="sdd|${DATA_BASE_DIR}/SDD"
DATASET_CONFIG["visa"]="visa|${DATA_BASE_DIR}/visa"

K_SHOTS=(2 16 64)
GPUS=(0 1 2 3 4 5 6 7)

# Generate all tasks
declare -a ALL_TASKS=()
for dataset_name in "${!DATASET_CONFIG[@]}"; do
    IFS="|" read -r DATASET_NAME DATASET_PATH <<< "${DATASET_CONFIG[$dataset_name]}"
    for k_shot in "${K_SHOTS[@]}"; do
        FOLDER_NAME=$(basename "$DATASET_PATH")
        SAVE_PATH="${RESULT_DIR}/${FOLDER_NAME}/shot-${k_shot}"
        ALL_TASKS+=("$DATASET_NAME|$DATASET_PATH|$k_shot|$SAVE_PATH")
    done
done

# Add full-shot tasks for sdd, dagm, dtd
for dataset_name in "sdd" "dagm" "dtd"; do
    IFS="|" read -r DATASET_NAME DATASET_PATH <<< "${DATASET_CONFIG[$dataset_name]}"
    FOLDER_NAME=$(basename "$DATASET_PATH")
    SAVE_PATH="${RESULT_DIR}/${FOLDER_NAME}/shot-full"
    ALL_TASKS+=("$DATASET_NAME|$DATASET_PATH|0|$SAVE_PATH")
done

echo "Total tasks: ${#ALL_TASKS[@]}"

# Distribute tasks to GPUs (round-robin)
declare -a GPU_TASKS_0=()
declare -a GPU_TASKS_1=()
declare -a GPU_TASKS_2=()
declare -a GPU_TASKS_3=()
declare -a GPU_TASKS_4=()
declare -a GPU_TASKS_5=()
declare -a GPU_TASKS_6=()
declare -a GPU_TASKS_7=()

for i in "${!ALL_TASKS[@]}"; do
    task="${ALL_TASKS[$i]}"
    gpu_idx=$(( i % 8 ))
    case $gpu_idx in
        0) GPU_TASKS_0+=("$task") ;;
        1) GPU_TASKS_1+=("$task") ;;
        2) GPU_TASKS_2+=("$task") ;;
        3) GPU_TASKS_3+=("$task") ;;
        4) GPU_TASKS_4+=("$task") ;;
        5) GPU_TASKS_5+=("$task") ;;
        6) GPU_TASKS_6+=("$task") ;;
        7) GPU_TASKS_7+=("$task") ;;
    esac
done

# Print task distribution
for i in 0 1 2 3 4 5 6 7; do
    count=$(eval "echo \${#GPU_TASKS_${i}[@]}")
    echo "GPU $i tasks: $count"
done

# Cleanup function
cleanup() {
    echo "Killing all training processes..."
    pkill -9 -P $$ 2>/dev/null || true
    exit 130
}
trap cleanup SIGINT SIGTERM

# Function to run GPU tasks
run_gpu_tasks() {
    local gpu_id=$1
    shift
    local tasks=("$@")
    
    for task in "${tasks[@]}"; do
        IFS="|" read -r DATASET_NAME DATASET_PATH K_SHOT SAVE_PATH <<< "$task"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu_id: Starting $DATASET_NAME (k=$K_SHOT)"
        
        python3 train.py \
            --train "$DATASET_NAME" \
            --train_set_path "$DATASET_PATH" \
            --gpu "$gpu_id" \
            --epoch 5 \
            --batch_size 16 \
            --k_shot "$K_SHOT" \
            --save_path "$SAVE_PATH" >> "$LOG_FILE" 2>&1
        
        if [ $? -ne 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu_id: FAILED $DATASET_NAME (k=$K_SHOT)"
            return 1
        fi
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu_id: Completed $DATASET_NAME (k=$K_SHOT)"
    done
    return 0
}

# Export function for subshells
export -f run_gpu_tasks
export LOG_FILE PROJECT_DIR

# Launch GPU workers
declare -a PIDS=()

run_gpu_tasks 0 "${GPU_TASKS_0[@]}" &
PIDS+=($!)

run_gpu_tasks 1 "${GPU_TASKS_1[@]}" &
PIDS+=($!)

run_gpu_tasks 2 "${GPU_TASKS_2[@]}" &
PIDS+=($!)

run_gpu_tasks 3 "${GPU_TASKS_3[@]}" &
PIDS+=($!)

run_gpu_tasks 4 "${GPU_TASKS_4[@]}" &
PIDS+=($!)

run_gpu_tasks 5 "${GPU_TASKS_5[@]}" &
PIDS+=($!)

run_gpu_tasks 6 "${GPU_TASKS_6[@]}" &
PIDS+=($!)

run_gpu_tasks 7 "${GPU_TASKS_7[@]}" &
PIDS+=($!)

echo "All GPU workers launched. PIDs: ${PIDS[*]}"
echo "Waiting for all workers to complete..."

# Wait for all
FAILED=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    wait "$pid"
    if [ $? -ne 0 ]; then
        echo "GPU worker $i (PID $pid) FAILED"
        FAILED=1
    else
        echo "GPU worker $i (PID $pid) completed"
    fi
done

if [ $FAILED -ne 0 ]; then
    echo "Some training tasks FAILED"
    exit 1
fi

echo "All Training Tasks Completed Successfully!"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
TRAIN_EOF

chmod +x train_all_parallel_taiji.sh

# Run training
bash train_all_parallel_taiji.sh

TRAIN_EXIT_CODE=$?

echo ""
echo "===== Training finished! ====="
echo "End time: $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

echo ""
echo "=========================================="
echo "RareCLIP TaiJi task completed successfully!"
echo "=========================================="

exit 0
