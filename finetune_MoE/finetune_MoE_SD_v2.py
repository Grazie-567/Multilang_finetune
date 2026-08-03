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
from deepspeed.moe.layer import MoE
from deepspeed.utils import groups
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
    """获取DeepSpeed分配的本地rank"""
    return int(os.environ.get("LOCAL_RANK", 0))

def get_world_size():
    """获取总进程数"""
    return int(os.environ.get("WORLD_SIZE", 1))

# ==================== 1. DeepSpeed MoE 层封装 ====================

class DeepSpeedMoELayer(MoE):
    """
    使用DeepSpeed MoE类封装Whisper的FFN层
    继承自deepspeed.moe.layer.MoE
    """
    def __init__(
        self,
        config,
        num_experts: int = 8,
        top_k: int = 2,
        expert_capacity: Optional[int] = None,
        min_capacity: int = 4,
        noisy_gate_policy: Optional[str] = "Jitter",
    ):
        self.config = config
        self.hidden_size = config.d_model
        self.intermediate_size = config.encoder_ffn_dim if hasattr(config, 'encoder_ffn_dim') else config.d_model * 4
        
        # 创建单个Expert（FFN）
        expert = nn.Sequential(
            nn.Linear(self.hidden_size, self.intermediate_size),
            nn.GELU(),
            nn.Linear(self.intermediate_size, self.hidden_size),
            nn.Dropout(config.dropout)
        )
        
        # 初始化DeepSpeed MoE层
        super().__init__(
            hidden_size=self.hidden_size,
            expert=expert,
            num_experts=num_experts,
            ep_size=1,  # 专家并行规模（由DeepSpeed自动管理）
            k=top_k,
            capacity_factor=expert_capacity,
            eval_capacity_factor=expert_capacity,
            min_capacity=min_capacity,
            noisy_gate_policy=noisy_gate_policy,
            use_tutel=False,  # 可根据需求启用Tutel优化
        )
        
        self.router_aux_loss_coef = 0.01  # 路由辅助损失系数

    def forward(self, hidden_states, **kwargs):
        """
        适配Whisper的调用接口
        """
        # DeepSpeed MoE需要2D输入 [batch*seq_len, hidden_size]
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)
        
        # 调用DeepSpeed MoE的前向传播
        output, gate_loss, _ = super().forward(hidden_states, **kwargs)
        
        # 恢复3D形状
        output = output.reshape(batch_size, seq_len, hidden_dim)
        
        # 返回格式与Whisper兼容
        return output, None, gate_loss  # (hidden_states, router_logits, aux_loss)


# ==================== 2. 模型MoE改造工具函数 ====================

def replace_ffn_with_deepspeed_moe(model, config, moe_config):
    """
    将指定层的FFN替换为DeepSpeed MoE层
    
    Args:
        model: Whisper模型
        config: 模型配置
        moe_config: MoE配置字典
    """
    num_experts = moe_config.get("num_experts", 8)
    top_k = moe_config.get("top_k", 2)
    moe_layers = moe_config.get("moe_layers", list(range(24, 32)))
    moe_decoder_layers = moe_config.get("moe_decoder_layers", list(range(30, 32)))
    # expert_capacity = moe_config.get("expert_capacity", 1.0)
    expert_capacity = 1
    
    encoder_layers = model.model.encoder.layers
    decoder_layers = model.model.decoder.layers
    
    replaced_count = 0
    
    def create_moe_wrapper(original_layer, expert_count, top_k_experts):
        """创建MoE包装器"""
        # 保存原组件
        fc1 = original_layer.fc1
        fc2 = original_layer.fc2
        activation = original_layer.activation_fn
        
        # 创建专家模块（复制原FFN结构）
        expert_module = nn.Sequential(
            fc1,
            activation,
            fc2,
            nn.Dropout(config.dropout)
        )
        
        # 创建MoE层
        moe = MoE(
            hidden_size=config.d_model,
            expert=expert_module,
            num_experts=expert_count,
            ep_size=1,
            k=top_k_experts,
            capacity_factor=expert_capacity,
            eval_capacity_factor=expert_capacity,
            min_capacity=4,
            noisy_gate_policy="Jitter",
            use_tutel=False,
        )
        
        # ===== 新增 shared expert =====
        shared_expert = nn.Sequential(
            nn.Linear(config.d_model, config.encoder_ffn_dim),
            nn.GELU(),
            nn.Linear(config.encoder_ffn_dim, config.d_model),
            nn.Dropout(config.dropout)
        )

        # ===== 权重初始化为原FFN =====
        with torch.no_grad():

            # fc1 → shared_expert[0]
            shared_expert[0].weight.copy_(fc1.weight)
            shared_expert[0].bias.copy_(fc1.bias)

            # fc2 → shared_expert[2]
            shared_expert[2].weight.copy_(fc2.weight)
            shared_expert[2].bias.copy_(fc2.bias)
        
        shared_alpha = nn.Parameter(torch.tensor(2.0))
        
        layer_expert_bias = nn.Parameter(torch.zeros(num_experts, config.d_model))
        with torch.no_grad():
            for i in range(num_experts):
                layer_expert_bias[i] = (i - num_experts // 2) * 0.01
        
        return moe, shared_expert, shared_alpha, layer_expert_bias
    
    # 替换编码器层的FFN
    for idx in moe_layers:
        if idx >= len(encoder_layers):
            continue
        
        layer = encoder_layers[idx]
        if not hasattr(layer, 'fc1'):
            print(f"⚠️  编码器层{idx}没有fc1，跳过")
            continue
        
        # 创建MoE并替换
        layer.moe_layer, layer.shared_expert, layer.shared_alpha, layer.expert_bias = create_moe_wrapper(layer, num_experts, top_k)
        # with torch.no_grad():
            # for p in layer.shared_expert.parameters():
                # p.lr_scale = 0.3
        
        # 删除原层（可选）
        del layer.fc1, layer.fc2
        
        # 修改forward
        def moe_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
        ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]]:
            """
            Whisper Encoder Layer with MoE FFN
            必须与原始WhisperEncoderLayer的返回签名完全一致
            """
            # 1. Self-Attention（保持不变）
            residual = hidden_states
            hidden_states = self.self_attn_layer_norm(hidden_states)
            
            # ✅ 注意：self_attn返回3个值 (hidden_states, attn_weights, present_key_value)
            hidden_states, attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            
            # 2. MoE Feed-Forward（替换原来的fc1/fc2）
            residual = hidden_states
            hidden_states = self.final_layer_norm(hidden_states)
            
            # 调整维度 for DeepSpeed MoE (需要2D输入)
            batch_size, seq_len, hidden_dim = hidden_states.shape
            hidden_states_2d = hidden_states.reshape(-1, hidden_dim)
            
            # ✅ DeepSpeed MoE前向
            # 返回: (output, gate_loss, expert_counts)
            
            # hidden_states_2d, gate_loss, _ = self.moe_layer(hidden_states_2d)
            # 新增shared expert
            moe_output, gate_loss, _ = self.moe_layer(hidden_states_2d)
            
            # ===== 加入 expert diversity =====
            bias = self.expert_bias.mean(dim=0)   # minimal侵入：不改router
            moe_output = moe_output + bias
            
            shared_output = self.shared_expert(hidden_states_2d)
            
            # 逐渐抑制shared expert的作用
            alpha = torch.sigmoid(self.shared_alpha)
            hidden_states_2d = moe_output + alpha * shared_output
            
            # 恢复维度
            hidden_states = hidden_states_2d.reshape(batch_size, seq_len, hidden_dim)
            
            # ✅ 将gate loss存储在layer中，供外部Trainer访问
            # 注意：必须保留这个属性，否则gate loss会被丢弃
            self._moe_gate_loss = gate_loss if self.training else None
            
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            
            # ✅ 3. 返回值格式必须与原始Whisper完全一致
            # 返回元组：(hidden_states, attn_weights, present_key_value) 或 (hidden_states,)
            # 即使output_attentions=False，也必须返回3个元素，后两个可以是None
            if output_attentions:
                return (hidden_states, attn_weights, present_key_value)
            else:
                # ⚠️ 关键点：必须返回3个值，后两个为None
                return (hidden_states, None, None)
        
        layer.forward = moe_forward.__get__(layer, type(layer))
        replaced_count += 1
        print(f"✅ 编码器层{idx}已替换为MoE")
    
    # 替换解码器层的FFN
    for idx in moe_decoder_layers:

        if idx >= len(decoder_layers):
            continue

        layer = decoder_layers[idx]
        if not hasattr(layer, 'fc1'):
            print(f"⚠️  解码器层{idx}没有fc1，跳过")
            continue

        layer.moe_layer, layer.shared_expert, layer.shared_alpha, layer.expert_bias = create_moe_wrapper(layer, num_experts, top_k)
        # with torch.no_grad():
            # for p in layer.shared_expert.parameters():
                # p.lr_scale = 0.3
        
        del layer.fc1, layer.fc2

        def decoder_moe_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = True,
            cache_position: Optional[torch.Tensor] = None,
        ):

            residual = hidden_states
            hidden_states = self.self_attn_layer_norm(hidden_states)

            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )

            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

            # ===== cross attention =====
            if encoder_hidden_states is not None:

                residual = hidden_states
                hidden_states = self.encoder_attn_layer_norm(hidden_states)

                hidden_states, cross_attn_weights, _ = self.encoder_attn(
                    hidden_states=hidden_states,
                    key_value_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    layer_head_mask=cross_attn_layer_head_mask,
                    output_attentions=output_attentions,
                )

                hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
                hidden_states = residual + hidden_states

            # ===== MoE FFN =====
            residual = hidden_states
            hidden_states = self.final_layer_norm(hidden_states)

            batch_size, seq_len, hidden_dim = hidden_states.shape
            hidden_states_2d = hidden_states.reshape(-1, hidden_dim)

            # 新增shared expert
            moe_output, gate_loss, _ = self.moe_layer(hidden_states_2d)
            
            # ===== 加入 expert diversity =====
            bias = self.expert_bias.mean(dim=0)   # minimal侵入：不改router
            moe_output = moe_output + bias
            
            shared_output = self.shared_expert(hidden_states_2d)
            
            # 逐渐抑制shared expert的作用
            alpha = torch.sigmoid(self.shared_alpha)
            hidden_states_2d = moe_output + alpha * shared_output

            hidden_states = hidden_states_2d.reshape(batch_size, seq_len, hidden_dim)

            self._moe_gate_loss = gate_loss if self.training else None

            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

            if output_attentions:
                return (hidden_states, self_attn_weights, present_key_value)
            else:
                return (hidden_states, None, None)

        layer.forward = decoder_moe_forward.__get__(layer, type(layer))

        replaced_count += 1
        print(f"✅ 解码器层{idx}已替换为MoE")
    
    print(f"🔄 共替换 {replaced_count} 层为DeepSpeed MoE架构")
    return model


# ==================== 3. 自定义Trainer（处理MoE损失） ====================

class DeepSpeedMoETrainer(Seq2SeqTrainer):
    """
    支持DeepSpeed MoE的Trainer，自动处理Gate Loss
    """
    def __init__(self, probs, teacher_model=None, old_language="en", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.moe_loss_scale = 0.01  # MoE辅助损失权重
        self._first_norm_logged = False  # 日志标记
        self.teacher_model = teacher_model
        self.old_language = old_language.split(",")
        self.probs = probs
        # ===== distillation weights =====
        self.distill_lambda = 0.5      # logit distillation
        self.hidden_lambda = 0.1       # hidden state distillation
        self.lang_lambda = 0.1         # language token distillation

        self.temperature = 2.0
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
    
    def compute_loss(
        self, 
        model, 
        inputs, 
        return_outputs=False, 
        **kwargs
        ):
        """
        重写损失计算，整合DeepSpeed MoE的Gate Loss
        
        处理逻辑：
        1. DeepSpeed自动前向传播并计算gate loss
        2. gate loss已自动添加到outputs.loss中
        3. 本方法仅处理Transformers 4.40.0+的num_items_in_batch归一化
        """
        # DeepSpeed会自动处理MoE的前向传播和损失计算
        # 只需确保output_router_logits=True
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
        
        # 主任务损失
        loss = outputs.loss
        
        # ✅ 捕获MoE gate loss（如果存在）
        # DeepSpeed会将gate loss存储在module._moe_gate_loss中
        if hasattr(model, "module"):
            base_model = model.module
        else:
            base_model = model
        
        total_gate_loss = 0.0
        num_moe_layers = 0
        # 遍历所有MoE层
        for layer in base_model.model.encoder.layers:
            if hasattr(layer, "_moe_gate_loss") and layer._moe_gate_loss is not None:
                total_gate_loss += layer._moe_gate_loss
                num_moe_layers += 1
        
        for layer in base_model.model.decoder.layers:
            if hasattr(layer, "_moe_gate_loss") and layer._moe_gate_loss is not None:
                total_gate_loss += layer._moe_gate_loss
                num_moe_layers += 1
        # 归一化
        if num_moe_layers > 0:
            total_gate_loss = total_gate_loss / num_moe_layers
        
        progress = self.state.global_step / self.state.max_steps
        progress = min(progress, 1.0)
        gate_weight = 0.01 + 0.04 * progress   # 0.01 → 0.05
        # 将gate loss添加到主loss（如果DeepSpeed没有自动添加）
        if total_gate_loss > 0 and not torch.isnan(total_gate_loss):
            loss += total_gate_loss * gate_weight  # 0.05是gate loss权重
            
        # if int(os.environ.get("RANK", 0)) == 0 and self.state.global_step % 50 == 0:
            # print(f"🧠 Gate loss: {total_gate_loss.item():.6f}")
        
        num_items_in_batch = kwargs.get("num_items_in_batch")
        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch * self.args.train_batch_size
            # 调试日志（仅打印一次）
            if self._first_norm_logged == False and int(os.environ.get("RANK", 0)) == 0:
                print(f"📊 损失归一化已激活: {num_items_in_batch} → {self.args.train_batch_size}")
                self._first_norm_logged = True
        
        # shared expert 衰减系数更新
        shared_reg_loss = 0.0

        for layer in base_model.model.encoder.layers:
            if hasattr(layer, "shared_alpha"):
                shared_reg_loss += torch.sigmoid(layer.shared_alpha)

        for layer in base_model.model.decoder.layers:
            if hasattr(layer, "shared_alpha"):
                shared_reg_loss += torch.sigmoid(layer.shared_alpha)

        loss += 0.001 * shared_reg_loss
        
        # distillation loss
        if teacher_logits is not None:
            
            distill_lambda = self.distill_lambda * (1 - progress)
            hidden_lambda = self.hidden_lambda * (1 - progress)
            lang_lambda = self.lang_lambda * (1 - progress)
            
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

            # ===== hidden state distillation =====
            student_hidden = outputs.decoder_hidden_states[-1][:, :min_len, :]
            teacher_hidden = teacher_outputs.decoder_hidden_states[-1][:, :min_len, :]

            hidden_loss = torch.nn.functional.mse_loss(
                student_hidden,
                teacher_hidden
            )

            # ===== language token distillation =====
            # Whisper token structure:
            # 0: <|startoftranscript|>
            # 1: <|language|>
            # 2: <|transcribe|>

            if student_hidden.shape[1] > 1:

                student_lang = student_hidden[:, 1, :]
                teacher_lang = teacher_hidden[:, 1, :]

                lang_loss = torch.nn.functional.mse_loss(
                    student_lang,
                    teacher_lang
                )

            else:

                lang_loss = torch.tensor(0.0, device=loss.device)

            # ===== final distillation loss =====
            distill_total = (
                distill_lambda * logit_loss
                + hidden_lambda * hidden_loss
                + lang_lambda * lang_loss
            )

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


# ==================== 4. 分布式初始化 ====================

def init_distributed():
    """
    初始化DeepSpeed分布式环境并打印GPU分配信息
    """
    if not torch.distributed.is_initialized():
        deepspeed.init_distributed()
    
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # 打印GPU分配信息
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(local_rank)
        print(f"🏁 [Global Rank {rank}/{world_size}] [Local Rank {local_rank}] "
              f"→ GPU-{local_rank}: {gpu_name}")
    
    # 同步所有进程
    torch.distributed.barrier()
    if rank == 0:
        print("✅ 所有进程的GPU分配验证完成！\n")
    
    return rank, world_size, local_rank  # 增加返回local_rank


# ==================== 5. 数据整理器 ====================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    max_label_length: int = 444  # ✅ 添加最大长度限制

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 截断过长的标签
        for f in features:
            if len(f["labels"]) > self.max_label_length:
                f["labels"] = f["labels"][:self.max_label_length]
        
        # 原有代码保持不变
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
        return batch


# ==================== 6. 主训练流程 ====================

def load_model_and_processor(model_path: str, language: str, task: str, moe_config=None):
    
    """加载模型和处理器，并应用DeepSpeed MoE改造"""
    from transformers import WhisperConfig
    
    if torch.cuda.is_available():
        gpu_capability = torch.cuda.get_device_capability()
        if gpu_capability[0] < 8:  # Ampere架构之前（A100, RTX30xx+才支持bf16）
            print("⚠️ 警告：当前GPU不支持bf16，将自动降级为fp16")
            use_bf16 = False
        else:
            use_bf16 = True
            print("✅ 检测到Ampere+ GPU，启用bf16模式")
    
    # 加载配置
    config = WhisperConfig.from_pretrained(model_path)
    
    # 加载模型
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,  # 使用FP16节省显存
    )
    
    # 应用MoE改造
    if moe_config is not None:
        print("🔄 正在应用DeepSpeed MoE架构改造...")
        model = replace_ffn_with_deepspeed_moe(
            model=model,
            config=config,
            moe_config=moe_config
        )
        config.moe_config = moe_config
        print("✅ DeepSpeed MoE改造完成")
    
    # 加载处理器
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
        fp16=not use_bf16,  # fp16与bf16互斥
        bf16=use_bf16,      # 启用bf16
        tf32=True if use_bf16 else False,  # Ampere+可启用tf32加速matmul
        dataloader_num_workers=8,
        per_device_eval_batch_size=args.eval_batch_size,
        generation_max_length=args.max_new_tokens,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        label_names=["labels"],
        push_to_hub=False,
        weight_decay=0.01,
        
        # DeepSpeed关键参数
        deepspeed=args.deepspeed,  # 指定ds_config.json路径
        ddp_find_unused_parameters=False,  # DeepSpeed会自动处理
        dataloader_drop_last=True,  # 确保批量大小一致
        
        # ZeRO-3优化
        optim=args.optim,  # 或"adamw_deepspeed"
    )


if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser(description="Whisper DeepSpeed MoE 微调")
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
    parser.add_argument("--output_dir", type=str, default="/mnt/lv2/FLEURS/whisper-finetune-results/MoE")
    parser.add_argument("--old_language", type=str, default="en")
    
    # DeepSpeed配置
    parser.add_argument("--deepspeed", type=str, default="ds_config_MoE.json", 
                       help="DeepSpeed配置文件路径")
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="DeepSpeed自动设置的本地rank")
    parser.add_argument("--optim", type=str, default="adamw_torch_fused",
                       help="优化器类型（bf16推荐adamw_torch_fused）")
    parser.add_argument("--tf32", action="store_true", default=True,
                       help="启用tf32加速（Ampere+）")
    
    # MoE配置
    parser.add_argument("--num_experts", type=int, default=8, help="专家数量")
    parser.add_argument("--top_k", type=int, default=2, help="Top-K路由")
    parser.add_argument("--moe_layers", type=str, default="64",
                       help="编码器MoE层索引")
    parser.add_argument("--moe_decoder_layers", type=str, default="",
                       help="解码器MoE层索引（可选）")
    parser.add_argument("--expert_capacity", type=int, default=None,
                       help="专家容量限制（None表示自动）")
    
    args = parser.parse_args()
    
    # 初始化分布式环境
    rank, world_size, local_rank = init_distributed()
    
    # 设置当前GPU（关键）
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    
     # 验证分布式loss记录配置
    if rank == 0:
        print("✅ LossTracker配置：主进程将负责记录和绘图")
        print(f"   输出目录: {args.output_dir}")
    else:
        print(f"⏭️  Rank {rank} 将跳过loss记录")
    
    # 解析MoE层配置
    moe_layers = [int(x) for x in args.moe_layers.split(",")] if args.moe_layers else []
    moe_decoder_layers = [int(x) for x in args.moe_decoder_layers.split(",")] if args.moe_decoder_layers else []
    
    moe_config = {
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "moe_layers": moe_layers,
        "moe_decoder_layers": moe_decoder_layers,
        "expert_capacity": args.expert_capacity,
        "router_aux_loss_coef": 0.05,
    }
    
    # 加载模型和处理器
    model, processor, use_bf16 = load_model_and_processor(
        args.model_path,
        args.language,
        args.task,
        moe_config=moe_config
    )
    
    teacher_model = load_teacher_model(args.model_path, local_rank)
    
    model.gradient_checkpointing_disable()
    
    # 打印MoE模型信息
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📊 模型总参数量: {total_params / 1e9:.2f}B")
        print(f"📊 可训练参数量: {trainable_params / 1e9:.2f}B")
        
        # 估算MoE激活参数量
        # ✅ 获取模型配置参数
        hidden_size = model.config.d_model
        intermediate_size = getattr(model.config, 'encoder_ffn_dim', hidden_size * 4)
        
        # ✅ 计算MoE相关参数量
        num_moe_layers = len(moe_layers) + len(moe_decoder_layers)
        experts_total_params = num_moe_layers * args.num_experts * intermediate_size * hidden_size * 2
        active_expert_params = num_moe_layers * args.top_k * intermediate_size * hidden_size * 2
        
        print(f"📊 MoE层数: {num_moe_layers}")
        print(f"📊 每个专家参数量: {(intermediate_size * hidden_size * 2) / 1e6:.2f}M")
        print(f"📊 总专家参数量: {experts_total_params / 1e9:.2f}B")
        print(f"📊 激活参数量（top-{args.top_k}）: {active_expert_params / 1e9:.2f}B")
    
    # 加载数据
    main_data_path = args.main_data_path
    ds = load_from_disk(main_data_path)
    
    # 配置训练参数
    training_args = setup_training_args(args.output_dir, args, use_bf16)
    # 配置loss记录器
    loss_tracker = LossTrackerCallback(
        output_dir=args.output_dir,
        plot_after_train=True,
        )
    
    teacher_model = teacher_model.to(torch.bfloat16)
    teacher_model.config.use_cache = False
    teacher_model.eval()
    
    # 高WER语种提升选择概率
    probs = [20.59, 16.02, 19.03, 17.09]
    # 初始化DeepSpeed MoE Trainer
    trainer = DeepSpeedMoETrainer(
        args=training_args,
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            max_label_length=444
            ),
        processing_class=processor,
        callbacks=[loss_tracker],
        teacher_model=teacher_model,
        old_language=args.old_language,
        probs=probs,
    )
    
    # 开始训练
    model.config.use_cache = False
    trainer.train()
    
    # 保存模型（仅在rank 0保存，避免重复）
    if rank == 0:
        model.config.moe_config = moe_config
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print(f"✅ 模型已保存至 {args.output_dir}")