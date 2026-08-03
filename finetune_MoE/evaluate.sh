#!/bin/bash

deepspeed --include="localhost:3" \
    --master_port=29501 \
    evaluate_wer_new.py \
    --model_path "/mnt/lv2/FLEURS/whisper-finetune-results/MoE/SD_v4_new_topk1_1e-6" \
    --save_path "/home/aiyuchen/MultiLang-Finetune/finetune_MoE/eval_results/SD_v4_new_topk1_1e-6" \
    --batch_size 16