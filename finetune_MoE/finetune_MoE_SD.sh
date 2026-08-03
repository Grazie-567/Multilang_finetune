#!/bin/bash

export NVIDIA_TF32_OVERRIDE=1

# 配置区域
MODEL_PATH="/mnt/lv2/FLEURS/whisper-large-v3"
DATA_PATH="/mnt/lv2/FLEURS/cache/main_data_large_new"
OUTPUT_DIR="/mnt/lv2/FLEURS/whisper-finetune-results/MoE/SD_v4_new_topk1_1e-6"
DEEPSPEED_CONFIG="/mnt/g2/aiyuchen/MultiLang-Finetune/finetune_MoE/ds_config_MoE.json"

# GPU选择（根据需求修改）
# 格式: localhost:GPU索引列表
# 示例: "localhost:0,1,2,3" 或 "localhost:0,2,4,6" 或 "node1:0-3@node2:0-3"
INCLUDE_SPEC="localhost:4,5,6,7"
OLD_LANGUAGE="ms,tl,id,vi"

# MoE配置
NUM_EXPERTS=8
TOP_K=1
MOE_LAYERS="24,25,26,27,28,29,30,31"
MOE_DECODER_LAYERS="30,31"

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

# 启动训练
echo "🚀 启动DeepSpeed MoE训练..."
echo "📊 GPU配置: $INCLUDE_SPEC"
echo "🎯 MoE编码器层: $MOE_LAYERS"
echo "🎯 MoE解码器层: $MOE_DECODER_LAYERS"
echo "🏗️  专家数: $NUM_EXPERTS"

deepspeed --include="$INCLUDE_SPEC" \
    --master_port=29000 \
    finetune_MoE_SD_v4.py \
    --model_path "$MODEL_PATH" \
    --main_data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --num_experts "$NUM_EXPERTS" \
    --top_k "$TOP_K" \
    --moe_layers "$MOE_LAYERS" \
    --moe_decoder_layers "$MOE_DECODER_LAYERS" \
    --train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-6 \
    --num_train_epochs 2 \
    --save_steps 1000 \
    --eval_steps 100 \
    --logging_steps 50 \
    --expert_capacity 1 \
    --old_language "$OLD_LANGUAGE" \
    --bf16

echo "✅ 训练完成！模型保存在: $OUTPUT_DIR"