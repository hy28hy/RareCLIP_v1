#!/bin/bash

# ==========================================
# RareCLIP Few-Shot 重跑 - 只跑缺失的任务
# 使用 GPU 4,5,6,7
# ==========================================

source /data/chenxuwu/anaconda3/etc/profile.d/conda.sh
conda activate RareCLIP

RARECLIP_DIR="/data/chenxuwu/zihaowan_workplace/RareCLIP"
SAVE_BASE_PATH="${RARECLIP_DIR}/visa_results"
WEIGHT="${RARECLIP_DIR}/weights/visa_pretrained.pth"
GPUS=(4 5 6 7)
SEEDS=(0 1 2 3 4)
SHOTS=(2 16 64)

# 只包含缺失的数据集 x shot 组合（排除已完成的）
# 已完成: Brain_AD(0,2,16,64), btad(0,2,16,64), visa(2,16,64), mvtec(2,16), MPDD(0,2,16), Liver_AD(0,2,16)
# 缺失: 下面全部
MISSING_TASKS=(
    # 格式: "test_name|data_path|folder_name"
    "mvtec|/data/chenxuwu/zihaowan_workplace/dataset/mvtec|mvtec"           # 缺: shot-0, 64; 有: 2,16
    "visa|/data/chenxuwu/zihaowan_workplace/dataset/visa|visa"               # 缺: shot-0; 有: 2,16,64
    "mpdd|/data/chenxuwu/zihaowan_workplace/dataset/MPDD|MPDD"               # 缺: 64; 有: 0,2,16
    "btad|/data/chenxuwu/zihaowan_workplace/dataset/BTech_Dataset_transformed|BTech_Dataset_transformed"  # 实际全有，保险起见重跑64
    "brain_ad|/data/chenxuwu/zihaowan_workplace/dataset/MedAD/Brain_AD|Brain_AD"   # 实际全有，保险起见
    "liver|/data/chenxuwu/zihaowan_workplace/dataset/MedAD/Liver_AD|Liver_AD"     # 缺: 64; 有: 0,2,16
    "retina|/data/chenxuwu/zihaowan_workplace/dataset/MedAD/Retina_RESC_AD|Retina_RESC_AD"     # 缺: 2,16,64; 有: 0
    "retina|/data/chenxuwu/zihaowan_workplace/dataset/MedAD/Retina_OCT2017_AD|Retina_OCT2017_AD" # 缺: 2,16,64; 有: 0
    "colondb|/data/chenxuwu/zihaowan_workplace/dataset/CVC-ColonDB|CVC-ColonDB"   # 缺: 2,16,64; 有: 0
    "clinicdb|/data/chenxuwu/zihaowan_workplace/dataset/CVC-ClinicDB|CVC-ClinicDB" # 缺: 2,16,64; 有: 0
    "kvasir|/data/chenxuwu/zihaowan_workplace/dataset/Kvasir|Kvasir"             # 缺: 2,16,64; 有: 0
    "cvc-300|/data/chenxuwu/zihaowan_workplace/dataset/CVC-300|CVC-300"           # 缺: 2,16,64; 有: 0
)

NUM_GPUS=${#GPUS[@]}

echo "=========================================="
echo "RareCLIP Few-Shot 重跑 (缺失任务)"
echo "GPU: ${GPUS[*]}"
echo "Shots: ${SHOTS[*]} | Seeds: ${SEEDS[*]}"
echo "数据集数: ${#MISSING_TASKS[@]}"
echo "$(date)"
echo "=========================================="

gpu_worker() {
    local my_gpu=$1
    local my_id=$2
    cd "${RARECLIP_DIR}"

    local task_idx=0
    for item in "${MISSING_TASKS[@]}"; do
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
echo "🎉 重跑完成! $(date)"
echo "💾 结果: ${SAVE_BASE_PATH}"
echo "=========================================="
