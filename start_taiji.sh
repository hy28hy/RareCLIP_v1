#!/bin/bash
#
# TaiJi startup script for RareCLIP training
# This script is called by TaiJi when the task starts
#

set -e

echo "=========================================="
echo "RareCLIP TaiJi Startup Script"
echo "=========================================="

# Step 0: Clone or find RareCLIP code
echo ""
echo "===== Step 0: Getting RareCLIP code ====="

CODE_DIR="/workspace/RareCLIP_v1"

if [ -d "$CODE_DIR" ]; then
    echo "Code directory already exists: $CODE_DIR"
    cd "$CODE_DIR"
else
    echo "Cloning RareCLIP_v1 from git..."
    cd /workspace
    git clone https://github.com/hy28hy/RareCLIP_v1.git
    cd RareCLIP_v1
fi

pwd
echo "Current directory contents:"
ls -la *.py 2>/dev/null | head -10

if [ ! -f "train.py" ]; then
    echo "ERROR: train.py not found in current directory"
    exit 1
fi

echo "===== Code ready! ====="

# Step 1: Install dependencies
echo ""
echo "===== Step 1: Installing dependencies ====="
pip install -i https://mirrors.tencent.com/pypi/simple/ \
    einops ftfy opencv-python pandas Pillow regex \
    scikit-image scikit-learn tabulate tqdm timm modelscope

echo "===== Dependencies installed! ====="

# Step 2: Download dataset
echo ""
echo "===== Step 2: Downloading dataset from ModelScope ====="
python3 download_dataset.py

if [ $? -ne 0 ]; then
    echo "ERROR: Dataset download failed"
    exit 1
fi

echo "===== Dataset ready! ====="

# Step 3: Make training script executable
echo ""
echo "===== Step 3: Preparing training script ====="
chmod +x train_all_parallel_taiji.sh

# Step 4: Start training
echo ""
echo "===== Step 4: Starting RareCLIP training ====="
echo "Training script: train_all_parallel_taiji.sh"
echo "Start time: $(date)"

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
