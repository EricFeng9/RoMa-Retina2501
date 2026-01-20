#!/bin/bash
# RoMa 多模态配准训练脚本

# 配准模式：cffa, cfoct, octfa, cfocta
MODE="cffa"

# 实验名称
NAME="roma_${MODE}_v1"

# 训练参数
BATCH_SIZE=4
NUM_WORKERS=8
IMG_SIZE=512
VESSEL_SIGMA=6.0
MAX_EPOCHS=100
GPUS="0"

# 启动训练
python experiments/train_roma_multimodal.py \
    --mode ${MODE} \
    --name ${NAME} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --img_size ${IMG_SIZE} \
    --vessel_sigma ${VESSEL_SIGMA} \
    --max_epochs ${MAX_EPOCHS} \
    --gpus ${GPUS} \
    --accelerator gpu \
    --strategy ddp_find_unused_parameters_false

echo "训练完成！结果保存在: results/roma_${MODE}/${NAME}"
