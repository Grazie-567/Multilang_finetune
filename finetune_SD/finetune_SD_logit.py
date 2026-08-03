import os
import torch
import torch.nn as nn
import argparse
import json
from dataclasses import dataclass
from typing import Any, List, Dict, Union, Optional, Tuple
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    HfArgumentParser,
    TrainerCallback,
)
from datasets import load_from_disk
import deepspeed
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
from LossTrainerCallback import LossTrackerCallback
from typing import Optional
import random

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("GPU count:", torch.cuda.device_count())

def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))

def get_world_size():
    return int(os.environ.get("WORLD_SIZE", 1))


class SelfDistillationTrainer(Seq2SeqTrainer):
    def __init__(self, probs, teacher_model=None, old_language="en", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._first_norm_logged = False
        self.teacher_model = teacher_model
        self.old_language = old_language.split(",")
        self.probs = probs
        # ===== distillation weights =====
        self.distill_lambda = 0.5      # logit distillation

        self.temperature = 2.0
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
    
    def compute_loss(
        self, 
        model, 
        inputs, 
        return_outputs=False, 
        **kwargs
        ):

        if hasattr(model, "module"):
            base_model = model.module
        else:
            base_model = model
        
        outputs = model(
            input_features=inputs["input_features"],
            labels=inputs["labels"],
            output_hidden_states=True,
            return_dict=True,
        )
        
        # teacher distillation
        teacher_logits = None

        if self.teacher_model is not None:
            with torch.no_grad():
                decoder_prompt = self.build_teacher_decoder_input(inputs["labels"])
                teacher_outputs = self.teacher_model(
                    input_features=inputs["input_features"],
                    decoder_input_ids=decoder_prompt,
                    output_hidden_states=True,
                    return_dict=True
                )
                teacher_logits = teacher_outputs.logits
        
        loss = outputs.loss
        
        progress = self.state.global_step / self.state.max_steps
        progress = min(progress, 1.0)
        
        num_items_in_batch = kwargs.get("num_items_in_batch")
        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch * self.args.train_batch_size
            if self._first_norm_logged == False and int(os.environ.get("RANK", 0)) == 0:
                print(f"📊 Enabled loss normalizaion: {num_items_in_batch} → {self.args.train_batch_size}")
                self._first_norm_logged = True
        
        # distillation loss
        if teacher_logits is not None:
            
            distill_lambda = self.distill_lambda * (1 - progress)
            
            student_logits = outputs.logits

            min_len = min(student_logits.shape[1], teacher_logits.shape[1])

            student_logits = student_logits[:, :min_len, :]
            teacher_logits = teacher_logits[:, :min_len, :]

            T = self.temperature * (2 - progress)

            student_log_probs = torch.log_softmax(student_logits / T, dim=-1)
            teacher_probs = torch.softmax(teacher_logits / T, dim=-1)

            logit_loss = self.kl_loss(
                student_log_probs,
                teacher_probs
            ) * (T * T)

            # ===== final distillation loss =====
            distill_total = distill_lambda * logit_loss    

            # loss = loss + distill_total
            loss = loss * (1.0 + 0.2 * progress) + distill_total
        
        return (loss, outputs) if return_outputs else loss
    
    def build_teacher_decoder_input(self, labels):
        tokenizer = self.processing_class.tokenizer
        start_token = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        lang = random.choices(self.old_language, weights=self.probs, k=1)[0]
        # print("Sampled language: " + lang)
        lang_token = tokenizer.convert_tokens_to_ids(f"<|{lang}|>")
        transcribe_token = tokenizer.convert_tokens_to_ids("<|transcribe|>")

        B = labels.shape[0]

        decoder_input = torch.full(
            (B, 3),
            fill_value=0,
            dtype=torch.long,
            device=labels.device
        )

        decoder_input[:, 0] = start_token
        decoder_input[:, 1] = lang_token
        decoder_input[:, 2] = transcribe_token
        
        return decoder_input

def init_distributed():
    if not torch.distributed.is_initialized():
        deepspeed.init_distributed()
    
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(local_rank)
        print(f"🏁 [Global Rank {rank}/{world_size}] [Local Rank {local_rank}] "
              f"→ GPU-{local_rank}: {gpu_name}")
    
    # Synchronizing
    torch.distributed.barrier()
    if rank == 0:
        print("✅ Validated GPU assignment for all processes!\n")
    
    return rank, world_size, local_rank

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    max_label_length: int = 444

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
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
    from transformers import WhisperConfig
    
    if torch.cuda.is_available():
        gpu_capability = torch.cuda.get_device_capability()
        if gpu_capability[0] < 8:  
            print("⚠️ Warning: bf16 not supported on current GPU. Automatically adjusted to fp16.")
            use_bf16 = False
        else:
            use_bf16 = True
            print("✅ Ampere+ GPU detected. bf16 is enabled")
    
    config = WhisperConfig.from_pretrained(model_path)
    
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,  # 使用FP16节省显存
    )
    
    processor = WhisperProcessor.from_pretrained(model_path, language=language, task=task)
    
    return model, processor, use_bf16

def load_teacher_model(model_path, device):
    teacher = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
        
    teacher.to(device)

    return teacher

def setup_training_args(output_dir: str, args, use_bf16: bool):
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
        fp16=not use_bf16,  
        bf16=use_bf16,      
        tf32=True if use_bf16 else False,
        dataloader_num_workers=8,
        per_device_eval_batch_size=args.eval_batch_size,
        generation_max_length=args.max_new_tokens,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        label_names=["labels"],
        push_to_hub=False,
        weight_decay=0.01,
        
        deepspeed=args.deepspeed,  # Appointing directory of deepspeed config
        ddp_find_unused_parameters=False,  
        dataloader_drop_last=True,
        
        # ZeRO-3 optim
        optim=args.optim,
    )

if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser(description="DeepSpeed Self-distillation FT")
    parser.add_argument("--model_path", type=str, default="/mnt/lv2/FLEURS/whisper-large-v3")
    parser.add_argument("--main_data_path", default="/mnt/lv2/FLEURS/cache/main_data_large_new")
    parser.add_argument("--task", type=str, default="transcribe")
    parser.add_argument("--language", type=str, default="id")
    parser.add_argument("--max_input_length", type=float, default=50.0)
    parser.add_argument("--num_test_samples", type=int, default=500)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--warmup_steps", type=int, default=250)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--max_new_tokens", type=int, default=444)
    parser.add_argument("--output_dir", type=str, default="/mnt/lv2/FLEURS/whisper-finetune-results/SD")
    parser.add_argument("--old_language", type=str, default="en")
    
    # DeepSpeed settings
    parser.add_argument("--deepspeed", type=str, default="ds_config.json", help="Directory of deepspeed config")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank automatically set by deepspeed")
    parser.add_argument("--optim", type=str, default="adamw_torch_fused", help="Type of optimizer")
    parser.add_argument("--tf32", action="store_true", default=True, help="Enable tf32 (for Ampere+)")
    
    args = parser.parse_args()
    
    # Initializing distributed environment
    rank, world_size, local_rank = init_distributed()
    
    # Setting current GPU device
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    
    if rank == 0:
        print("✅ LossTracker: Main process will record the loss and plot")
        print(f"output directory: {args.output_dir}")
    else:
        print(f"⏭️  Rank {rank} will skip loss record")
    
    # Loacding model and processor
    model, processor, use_bf16 = load_model_and_processor(args.model_path, args.language, args.task)
    teacher_model = load_teacher_model(args.model_path, local_rank)
    model.gradient_checkpointing_disable()
    
    # Loading dataset
    main_data_path = args.main_data_path
    ds = load_from_disk(main_data_path)
    
    # Setting training arguments
    training_args = setup_training_args(args.output_dir, args, use_bf16)
    
    # Setting loss tracker
    loss_tracker = LossTrackerCallback(
        output_dir=args.output_dir,
        plot_after_train=True,
    )
    
    teacher_model = teacher_model.to(torch.bfloat16)
    teacher_model.config.use_cache = False
    teacher_model.eval()
    
    # Weighted probability of old language choice
    probs = [20.59, 16.02, 19.03, 17.09]
    
    # Initializing trainer
    trainer = SelfDistillationTrainer(
        args = training_args,
        model = model,
        train_dataset = ds["train"],
        eval_dataset = ds["test"],
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor = processor,
            max_label_length = 444
        ),
        processing_class = processor,
        callbacks = [loss_tracker],
        teacher_model = teacher_model,
        old_language = args.old_language,
        probs = probs,
    )
    
    # Start training
    model.config.use_cache = False
    trainer.train()
    
    # Saving model only at rank 0
    if rank == 0:
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print(f"✅ The model has been saved at {args.output_dir}")