#!/bin/bash

export NVIDIA_TF32_OVERRIDE=1

# 配置区域
MODEL_PATH="/mnt/lv2/FLEURS/whisper-small"
DATA_PATH="/mnt/lv2/FLEURS/cache/main_data_med_new"
OUTPUT_DIR="/mnt/lv2/FLEURS/whisper-finetune-results/SD/small-analyze"
DEEPSPEED_CONFIG="/mnt/g2/aiyuchen/MultiLang-Finetune/finetune_SD/ds_config.json"

# GPU选择（根据需求修改）
# 格式: localhost:GPU索引列表
# 示例: "localhost:0,1,2,3" 或 "localhost:0,2,4,6" 或 "node1:0-3@node2:0-3"
INCLUDE_SPEC="localhost:2,3,4,5"
OLD_LANGUAGE="ms,tl,id,vi"

# 检查DeepSpeed
if ! command -v deepspeed &> /dev/null; then
    echo "错误: DeepSpeed未安装"
    exit 1
fi

# 检查配置文件
if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo "错误: DeepSpeed配置文件不存在: $DEEPSPEED_CONFIG"
    exit 1
fi

deepspeed --include="$INCLUDE_SPEC" \
    --master_port=29111 \
    finetune_SD_analyze_base.py \
    --model_path "$MODEL_PATH" \
    --main_data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --num_train_epochs 2 \
    --save_steps 1000 \
    --eval_steps 100 \
    --logging_steps 50 \
    --old_language "$OLD_LANGUAGE" \
    --bf16

echo "✅ 训练完成！模型保存在: $OUTPUT_DIR"