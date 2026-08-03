#!/usr/bin/env bash
# 说明：在工程根目录运行

# 如有需要，可自定义 master_port 避免端口冲突
MASTER_PORT=29010

deepspeed \
  --include="localhost:0,1,2,3" \
  --master_port=${MASTER_PORT} \
  finetune_noer.py \
  --deepspeed "/home/aiyuchen/MultiLang-Finetune/finetune/ds_config_new.json" \
  --model_path "/mnt/lv2/FLEURS/whisper-small" \
  --output_dir "/mnt/lv2/FLEURS/whisper-finetune-results/ALL/small" \
  --main_data_path "/mnt/lv2/FLEURS/cache/main_data_med" \
  --train_batch_size 2 \
  --eval_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --num_train_epochs 2 \
  --warmup_steps 500 \
  --save_steps 1000 \
  --eval_steps 100 \
  --logging_steps 50 \
  --max_new_tokens 444 \
  --bf16