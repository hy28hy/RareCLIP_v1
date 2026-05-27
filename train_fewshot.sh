#!/bin/bash

# ==========================================
# Few-shot 训练脚本
# 训练 k_shot = 2, 16, 64
# 支持样本不足时的重复采样
# ==========================================

GPU_ID=0
BASE_DATA_DIR="/data/chenxuwu/zihaowan_workplace/dataset"
RESULT_DIR="./fewshot_results"

# 创建结果目录
mkdir -p ${RESULT_DIR}

# 格式: "数据集名称(用于--train参数)|数据集真实物理路径|预训练权重"
# MVTec/Btad/MPDD/CVC-300/Retina 使用 visa 预训练权重
# Visa 使用 mvtec 预训练权重（因为 visa 没有预训练权重）

# 定义训练数据集
TRAIN_DATASETS=(
    "mvtec|${BASE_DATA_DIR}/mvtec|visa"
    "btad|${BASE_DATA_DIR}/BTech_Dataset_transformed|visa"
    "mpdd|${BASE_DATA_DIR}/MPDD|visa"
    "cvc-300|${BASE_DATA_DIR}/CVC-300|visa"
    "retina|${BASE_DATA_DIR}/MedAD/Retina_OCT2017_AD|visa"
    "visa|${BASE_DATA_DIR}/visa|mvtec"
    "dagm|${BASE_DATA_DIR}/DAGM/DAGM_KaggleUpload|visa"
    "dtd|${BASE_DATA_DIR}/DTD-Synthetic/DTD-Synthetic|visa"
    "sdd|${BASE_DATA_DIR}/SDD|visa"
)

# k-shot 值列表
K_SHOT_LIST=(2 16 64)

# 成功和失败记录
SUCCESS_LIST=()
FAIL_LIST=()

echo "================================================================"
echo "🚀 开始 Few-shot 训练"
echo "🎯 训练 k_shot: ${K_SHOT_LIST[@]}"
echo "📁 权重保存目录: ${RESULT_DIR}"
echo "================================================================"

for item in "${TRAIN_DATASETS[@]}"; do
    # 按照 "|" 分割名字、路径和预训练权重
    IFS="|" read -r DATA_NAME DATA_PATH PRETRAIN_WEIGHT <<< "$item"
    
    # 提取末尾的文件夹名用于保存结果的命名
    FOLDER_NAME=$(basename "$DATA_PATH")
    
    for k_shot in "${K_SHOT_LIST[@]}"; do
        echo ""
        echo "================================================================"
        echo "📌 训练配置"
        echo "   数据集: $DATA_NAME"
        echo "   k_shot: $k_shot"
        echo "   预训练权重: $PRETRAIN_WEIGHT"
        echo "   权重保存路径: ${RESULT_DIR}/${FOLDER_NAME}/shot-${k_shot}"
        echo "================================================================"

        # 执行训练命令
        python train.py \
            --train "$DATA_NAME" \
            --train_set_path "$DATA_PATH" \
            --gpu "$GPU_ID" \
            --epoch 5 \
            --batch_size 16 \
            --k_shot "$k_shot" \
            --save_path "${RESULT_DIR}/${FOLDER_NAME}/shot-${k_shot}"

        # 检查上一条命令是否成功运行并记录到对应数组中
        if [ $? -eq 0 ]; then
            echo ">> ${FOLDER_NAME}/shot-${k_shot} 训练完成。"
            SUCCESS_LIST+=("${FOLDER_NAME}/shot-${k_shot}")
        else
            echo ">> ${FOLDER_NAME}/shot-${k_shot} 训练中断或报错，跳过..."
            FAIL_LIST+=("${FOLDER_NAME}/shot-${k_shot}")
        fi
    done
done

# ==========================================
# 最终结果统计与输出
# ==========================================
echo ""
echo "================================================================"
echo "🎉 所有 Few-shot 训练完毕！最终统计结果如下："
echo "================================================================"

echo "✅ 成功的任务 (${#SUCCESS_LIST[@]} 个):"
if [ ${#SUCCESS_LIST[@]} -gt 0 ]; then
    for task in "${SUCCESS_LIST[@]}"; do
        echo "  - $task"
    done
else
    echo "  (无)"
fi

echo ""
echo "❌ 失败的任务 (${#FAIL_LIST[@]} 个):"
if [ ${#FAIL_LIST[@]} -gt 0 ]; then
    for task in "${FAIL_LIST[@]}"; do
        echo "  - $task"
    done
else
    echo "  🎈 完美！没有任何任务报错！"
fi
echo "================================================================"
