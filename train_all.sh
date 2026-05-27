#!/bin/bash

# ==========================================
# 配置区域
# ==========================================
GPU_ID=0  # 使用哪块显卡
BASE_DATA_DIR="/data/chenxuwu/zihaowan_workplace/dataset"

# 格式: "数据集名称(用于--train参数)|数据集真实物理路径"

# 1. 工业数据集
INDUSTRIAL_DATASETS=(
    "mpdd|${BASE_DATA_DIR}/MPDD"
    "btad|${BASE_DATA_DIR}/BTech_Dataset_transformed"
)

# 2. 器官类医疗数据集
MEDICAL_DATASETS=(
    # "brain_ad|${BASE_DATA_DIR}/MedAD/Brain_AD"
    # "liver|${BASE_DATA_DIR}/MedAD/Liver_AD"
    # "retina|${BASE_DATA_DIR}/MedAD/Retina_RESC_AD"
    "retina|${BASE_DATA_DIR}/MedAD/Retina_OCT2017_AD"
) 

# 3. 肠息肉类数据集
POLYP_DATASETS=(
    "colondb|${BASE_DATA_DIR}/CVC-ColonDB"
    "clinicdb|${BASE_DATA_DIR}/CVC-ClinicDB"
    "kvasir|${BASE_DATA_DIR}/Kvasir"
    "cvc-300|${BASE_DATA_DIR}/CVC-300"
)

# 将所有任务合并到一个数组中
ALL_DATASETS=("${MEDICAL_DATASETS[@]}" )

# ==========================================
# 准备记录成功与失败的数组
# ==========================================
SUCCESS_LIST=()
FAIL_LIST=()

# ==========================================
# 执行训练循环
# ==========================================
for item in "${ALL_DATASETS[@]}"; do
    # 按照 "|" 分割名字和路径
    IFS="|" read -r DATA_NAME DATA_PATH <<< "$item"
    
    # 提取末尾的文件夹名用于保存权重的命名
    FOLDER_NAME=$(basename "$DATA_PATH")
    
    echo "================================================================"
    echo "🚀 开始训练任务"
    echo "🏷️  参数名称: $DATA_NAME"
    echo "📂 数据路径: $DATA_PATH"
    echo "💾 保存路径: ./exps/${FOLDER_NAME}_train"
    echo "🕒 开始时间: $(date)"
    echo "================================================================"

    # 执行训练命令
    python train.py \
        --train "$DATA_NAME" \
        --train_set_path "$DATA_PATH" \
        --gpu "$GPU_ID" \
        --epoch 5 \
        --batch_size 16 \
        --save_path "./exps/${FOLDER_NAME}_train"

    # 检查上一条命令是否成功运行并记录到对应数组中
    if [ $? -eq 0 ]; then
        echo ">> $FOLDER_NAME 训练结束。"
        SUCCESS_LIST+=("$FOLDER_NAME")
    else
        echo ">> $FOLDER_NAME 训练中断或报错，跳过并执行下一个..."
        FAIL_LIST+=("$FOLDER_NAME")
    fi
    echo ""
done

# ==========================================
# 最终结果统计与输出
# ==========================================
echo "================================================================"
echo "🎉 所有数据集遍历完毕！最终统计结果如下："
echo "================================================================"

# 输出成功的列表
echo "✅ 成功的数据集 (${#SUCCESS_LIST[@]} 个):"
if [ ${#SUCCESS_LIST[@]} -gt 0 ]; then
    for ds in "${SUCCESS_LIST[@]}"; do
        echo "  - $ds"
    done
else
    echo "  (无)"
fi

echo ""

# 输出失败的列表
echo "❌ 失败的数据集 (${#FAIL_LIST[@]} 个):"
if [ ${#FAIL_LIST[@]} -gt 0 ]; then
    for ds in "${FAIL_LIST[@]}"; do
        echo "  - $ds"
    done
else
    echo "  🎈 完美！没有任何数据集报错！"
fi
echo "================================================================"