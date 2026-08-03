import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import torch
import argparse
from dataclasses import dataclass
from typing import Any, List, Dict, Union
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_from_disk, concatenate_datasets

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def load_model_and_processor(model_path: str, language: str, task: str):
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    processor = WhisperProcessor.from_pretrained(model_path, language=language, task=task)
    return model, processor

def setup_training_args(output_dir: str, args):
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        deepspeed="/home/aiyuchen/MultiLang-Finetune/finetune/ds_config_ALL.json",
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="steps",
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        fp16=False,
        bf16=True,
        dataloader_num_workers=8,
        per_device_eval_batch_size=args.eval_batch_size,
        generation_max_length=args.max_new_tokens,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        label_names=["labels"],
        push_to_hub=False,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper 全参数微调")
    parser.add_argument("--model_path", type=str, default="/mnt/lv2/FLEURS/whisper-large-v3")
    parser.add_argument("--processed_data_root",  default="/mnt/lv2/FLEURS/cache")
    parser.add_argument("--task", type=str, default="transcribe")
    parser.add_argument("--language", type=str, default="id")  # 用于 processor 配置
    parser.add_argument("--max_input_length", type=float, default=50.0)
    parser.add_argument("--num_test_samples", type=int, default=500)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=225)
    parser.add_argument("--output_dir", type=str, default="/mnt/lv2/FLEURS/whisper-finetune-results/ALL/large-v3")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--deepspeed", type=str)
    args = parser.parse_args()
    torch.cuda.empty_cache()

    # 加载模型和处理器
    model, processor = load_model_and_processor(args.model_path, args.language, args.task)

    main_data_path = os.path.join(args.processed_data_root, "main_data_large")
    er_data_path = os.path.join(args.processed_data_root, "er_data_large")
    ds = load_from_disk(main_data_path)
    ds_replay = load_from_disk(er_data_path)
    combined_train_data = concatenate_datasets([ds["train"], ds_replay["train"]]).shuffle(seed=42)
    combined_test_data = concatenate_datasets([ds["test"], ds_replay["test"]]).shuffle(seed=42)
    
    # 训练参数
    training_args = setup_training_args(args.output_dir, args)

    # 初始化 Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=combined_train_data,
        eval_dataset=combined_test_data,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor),
        tokenizer=processor.feature_extractor,
    )

    model.config.use_cache = False  # 防止 warning
    model.gradient_checkpointing_enable()
    trainer.train()

    # 保存模型和处理器
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
