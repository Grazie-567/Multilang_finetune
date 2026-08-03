#!/bin/bash

export NVIDIA_TF32_OVERRIDE=1

# 配置区域
MODEL_PATH="/mnt/lv2/FLEURS/whisper-large-v3"
DATA_PATH="/mnt/lv2/FLEURS/cache"
OUTPUT_DIR="/mnt/lv2/FLEURS/whisper-finetune-results/MoE/test_last2"
DEEPSPEED_CONFIG="/mnt/g2/aiyuchen/MultiLang-Finetune/finetune_MoE/ds_config_MoE.json"

# GPU选择（根据需求修改）
# 格式: localhost:GPU索引列表
# 示例: "localhost:0,1,2,3" 或 "localhost:0,2,4,6" 或 "node1:0-3@node2:0-3"
INCLUDE_SPEC="localhost:0,1,2,3"

# MoE配置
NUM_EXPERTS=8
TOP_K=2
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
    --master_port=29513 \
    finetune_MoE.py \
    --model_path "$MODEL_PATH" \
    --processed_data_root "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --num_experts "$NUM_EXPERTS" \
    --top_k "$TOP_K" \
    --moe_decoder_layers "$MOE_DECODER_LAYERS" \
    --train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --num_train_epochs 2 \
    --save_steps 500 \
    --eval_steps 50 \
    --logging_steps 10 \
    --expert_capacity 1 \
    --bf16

echo "✅ 训练完成！模型保存在: $OUTPUT_DIR"