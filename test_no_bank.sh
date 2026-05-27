#!/bin/bash

# ==========================================
# RareCLIP 无 Bank (Memory) 测试
# 禁用 PFM/PSM prototype feature memory bank
# 只用 Text Prompt + Image Rarity 分支
# 使用 GPU 0, 1, 2, 3
# ==========================================

source /data/chenxuwu/anaconda3/etc/profile.d/conda.sh
conda activate RareCLIP

RARECLIP_DIR="/data/chenxuwu/zihaowan_workplace/RareCLIP"
SAVE_BASE_PATH="${RARECLIP_DIR}/no_bank_results"
WEIGHT="${RARECLIP_DIR}/weights/visa_pretrained.pth"
GPUS=(0 1 2 3)
SHOTS=(0 2 16 64)
SEEDS=(0 1 2 3 4)

ALL_TASKS=(
    "mvtec|/data/chenxuwu/zihaowan_workplace/dataset/mvtec|mvtec"
    "visa|/data/chenxuwu/zihaowan_workplace/dataset/visa|visa"
    "mpdd|/data/chenxuwu/zihaowan_workplace/dataset/MPDD|MPDD"
    "btad|/data/chenxuwu/zihaowan_workplace/dataset/BTech_Dataset_transformed|BTech_Dataset_transformed"
    "brain_ad|/data/chenxuwu/zihaowan_workplace/dataset/MedAD/Brain_AD|Brain_AD"
    "liver|/data/chenxuwu/zihaowan_workplace/dataset/MedAD/Liver_AD|Liver_AD"
    "retina|/data/chenxuwu/zihaowan_workplace/dataset/MedAD/Retina_RESC_AD|Retina_RESC_AD"
    "retina|/data/chenxuwu/zihaowan_workplace/dataset/MedAD/Retina_OCT2017_AD|Retina_OCT2017_AD"
    "colondb|/data/chenxuwu/zihaowan_workplace/dataset/CVC-ColonDB|CVC-ColonDB"
    "clinicdb|/data/chenxuwu/zihaowan_workplace/dataset/CVC-ClinicDB|CVC-ClinicDB"
    "kvasir|/data/chenxuwu/zihaowan_workplace/dataset/Kvasir|Kvasir"
    "cvc-300|/data/chenxuwu/zihaowan_workplace/dataset/CVC-300|CVC-300"
)

NUM_GPUS=${#GPUS[@]}

echo "=========================================="
echo "RareCLIP No-Bank 测试 (无 Memory Bank)"
echo "GPU: ${GPUS[*]}"
echo "Shots: ${SHOTS[*]} | Seeds: ${SEEDS[*]}"
echo "数据集数: ${#ALL_TASKS[@]} | 总任务: $(( ${#ALL_TASKS[@]} * ${#SHOTS[@]} * ${#SEEDS[@]} ))"
echo "$(date)"
echo "=========================================="

gpu_worker() {
    local my_gpu=$1
    local my_id=$2
    cd "${RARECLIP_DIR}"

    local task_idx=0
    for item in "${ALL_TASKS[@]}"; do
        IFS="|" read -r DATA_NAME DATA_PATH FOLDER_NAME <<< "$item"

        for shot in "${SHOTS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                if [ $((task_idx % NUM_GPUS)) -ne $my_id ]; then
                    task_idx=$((task_idx + 1))
                    continue
                fi

                local save_path="${SAVE_BASE_PATH}/${FOLDER_NAME}/shot-${shot}"
                mkdir -p "$save_path"

                echo "[GPU$my_gpu] 🧪 $DATA_NAME ($FOLDER_NAME) shot-$shot seed-$seed 开始"

                CUDA_VISIBLE_DEVICES=$my_gpu python test.py \
                    --test "$DATA_NAME" \
                    --test_set_path "$DATA_PATH" \
                    --gpu 0 \
                    --seed $seed \
                    --k_shot $shot \
                    --load_path "$WEIGHT" \
                    --num_workers 8 \
                    --metric px+sp \
                    --no_bank 1 \
                    >> "$save_path/log.txt" 2>&1

                if [ $? -eq 0 ]; then
                    echo "[GPU$my_gpu] ✅ $DATA_NAME shot-$shot seed-$seed 完成"
                else
                    echo "[GPU$my_gpu] ❌ $DATA_NAME shot-$shot seed-$seed 失败"
                fi

                task_idx=$((task_idx + 1))
            done
        done
    done
    echo "[GPU$my_gpu] ===== All tasks done ====="
}

PIDS=()
for i in "${!GPUS[@]}"; do
    gpu_worker "${GPUS[$i]}" "$i" &
    PIDS+=($!)
    echo "Worker$i -> GPU${GPUS[$i]} (PID: ${PIDS[$i]})"
done

echo ""
echo "等待所有 Worker 完成..."
wait "${PIDS[@]}"

echo ""
echo "=========================================="
echo "🎉 No-Bank 测试完成! $(date)"
echo "💾 结果: ${SAVE_BASE_PATH}"
echo "=========================================="
