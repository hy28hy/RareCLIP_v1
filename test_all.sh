#!/bin/bash

# ==========================================
# 测试配置区域
# ==========================================
GPU_ID=0  # 测试使用的显卡
BASE_DATA_DIR="/data/chenxuwu/zihaowan_workplace/dataset"

# 格式: "数据集名称(用于--test参数)|数据集真实物理路径"
INDUSTRIAL_DATASETS=(
    "mpdd|${BASE_DATA_DIR}/MPDD"
    "btad|${BASE_DATA_DIR}/BTech_Dataset_transformed"
)

MEDICAL_DATASETS=(
    "brain_ad|${BASE_DATA_DIR}/MedAD/Brain_AD"
    "liver|${BASE_DATA_DIR}/MedAD/Liver_AD"
    "retina|${BASE_DATA_DIR}/MedAD/Retina_RESC_AD"
    "retina|${BASE_DATA_DIR}/MedAD/Retina_OCT2017_AD"
)

POLYP_DATASETS=(
    "colondb|${BASE_DATA_DIR}/CVC-ColonDB"
    "clinicdb|${BASE_DATA_DIR}/CVC-ClinicDB"
    "kvasir|${BASE_DATA_DIR}/Kvasir"
    "cvc-300|${BASE_DATA_DIR}/CVC-300"
)

ALL_DATASETS=(
                "${INDUSTRIAL_DATASETS[@]}"
                "${MEDICAL_DATASETS[@]}" 
                "${POLYP_DATASETS[@]}"
                )

# ==========================================
# 自动化测试循环
# ==========================================
for item in "${ALL_DATASETS[@]}"; do
    IFS="|" read -r DATA_NAME DATA_PATH <<< "$item"
    
    # 提取文件夹名，用于寻找对应的权重
    FOLDER_NAME=$(basename "$DATA_PATH")
    WEIGHT_PATH="/data/chenxuwu/zihaowan_workplace/RareCLIP/weights/visa_pretrained.pth"
    SAVE_RESULTS_PATH="./visa_results/${FOLDER_NAME}"

    echo "================================================================"
    echo "🧪 开始测试任务"
    echo "🏷️  数据集名: $DATA_NAME"
    echo "📂 测试路径: $DATA_PATH"
    echo "⚖️  权重路径: $WEIGHT_PATH"
    echo "================================================================"

    # 检查训练好的权重是否存在
    if [ ! -f "$WEIGHT_PATH" ]; then
        echo "⚠️ 警告：找不到权重文件 $WEIGHT_PATH ！跳过该数据集。"
        echo ""
        continue
    fi

    # 执行测试命令 (注意这里传入了 --load_path 加载你对应的pth)
    python test.py \
        --test "$DATA_NAME" \
        --test_set_path "$DATA_PATH" \
        --gpu "$GPU_ID" \
        --load_path "$WEIGHT_PATH" \
        --save_path "$SAVE_RESULTS_PATH" \
        --num_workers 8

    if [ $? -eq 0 ]; then
        echo "✅ $FOLDER_NAME 测试完成！结果已保存到 $SAVE_RESULTS_PATH"
    else
        echo "❌ $FOLDER_NAME 测试报错！"
    fi
    echo ""
done

echo "🎉 所有数据集测试任务已执行完毕！"