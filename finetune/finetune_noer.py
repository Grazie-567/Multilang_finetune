import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
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
from datasets import load_from_disk

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    max_label_length: int = 444
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 截断过长的标签
        for f in features:
            if len(f["labels"]) > self.max_label_length:
                f["labels"] = f["labels"][:self.max_label_length]
        
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

    # 修复 UserWarning：确保 generation_config 正确设置
    if model.generation_config is None or not getattr(model.generation_config, 'suppress_tokens', None):
        generation_config = GenerationConfig.from_model_config(model.config)
        model.generation_config = generation_config
    
    return model, processor

def setup_training_args(output_dir: str, args):
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="steps",
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        tf32=True if args.bf16 else False,
        dataloader_num_workers=8,
        per_device_eval_batch_size=args.eval_batch_size,
        generation_max_length=args.max_new_tokens,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        label_names=["labels"],
        push_to_hub=False,
        deepspeed=args.deepspeed,  # 指定ds_config.json路径
        ddp_find_unused_parameters=False,  # DeepSpeed会自动处理
        dataloader_drop_last=True,  # 确保批量大小一致
        weight_decay=0.01,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper 全参数微调")
    parser.add_argument("--model_path", type=str, default="/mnt/lv2/FLEURS/whisper-large-v3")
    parser.add_argument("--main_data_path",  default="/mnt/lv2/FLEURS/cache/large-v3")
    parser.add_argument("--task", type=str, default="transcribe")
    parser.add_argument("--language", type=str, default="id")  # 用于 processor 配置
    parser.add_argument("--max_input_length", type=float, default=50.0)
    parser.add_argument("--num_test_samples", type=int, default=500)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--warmup_steps", type=int, default=250)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--max_new_tokens", type=int, default=444)
    parser.add_argument("--output_dir", type=str, default="/mnt/lv2/FLEURS/whisper-finetune-results/ALL/large-v3-1e-4")
    parser.add_argument("--deepspeed", type=str)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--optim", type=str, default="adamw_torch_fused", help="优化器类型（bf16推荐adamw_torch_fused）")
    args = parser.parse_args()
    torch.cuda.empty_cache()

    # 加载模型和处理器
    model, processor = load_model_and_processor(args.model_path, args.language, args.task)

    main_data_path = args.main_data_path
    ds = load_from_disk(main_data_path)
    
    # 训练参数
    training_args = setup_training_args(args.output_dir, args)

    # 初始化 Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor),
        processing_class=processor,
    )

    model.config.use_cache = False  # 防止 warning
    trainer.train()

    # 保存模型和处理器
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
