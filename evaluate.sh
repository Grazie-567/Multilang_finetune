#!/bin/bash

deepspeed --include="localhost:7" \
    --master_port=29513 \
    evaluate_wer_new.py \
    --model_path "/mnt/lv2/FLEURS/whisper-finetune-results/SD/small" \
    --processor_path "/mnt/lv2/FLEURS/whisper-small" \
    --save_path "/home/aiyuchen/MultiLang-Finetune/finetune_SD/eval_results/small" \
    --batch_size 16