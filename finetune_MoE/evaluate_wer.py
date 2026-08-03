import os, json, csv, datetime
import torch
import glob
import numpy as np
import argparse
import gc
import jiwer
import pandas as pd
from safetensors.torch import load_file
from transformers import WhisperProcessor, WhisperConfig, WhisperForConditionalGeneration
from whisper.normalizers import BasicTextNormalizer
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from tqdm import tqdm
from load_datasets import load_process_datasets
import zhconv
from datasets import load_from_disk
from finetune_MoE import replace_ffn_with_deepspeed_moe
import deepspeed

# 2. 在所有 transformers 代码之前，设置这个环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_whisper_moe_model(model_path, moe_config=None):
    """
    从safetensors分片文件加载带有MoE结构的Whisper模型
    
    Args:
        model_path: 模型输出目录（包含safetensors文件）
        moe_config: MoE配置字典，如果为None则从config.json读取
    
    Returns:
        model: 带有MoE结构的模型
    """
    # 1. 加载config
    config = WhisperConfig.from_pretrained(model_path)
    
    # 2. 获取MoE配置（优先级：参数 > config.json）
    if moe_config is None:
        if hasattr(config, 'moe_config'):
            moe_config = config.moe_config
        else:
            # 硬编码你的MoE配置（如果config.json中没有）
            moe_config = {
                "num_experts": 8,
                "top_k": 2,
                "moe_layers": [24, 25, 26, 27, 28, 29, 30, 31],  # 你的MoE层
                "moe_decoder_layers": [30, 31],
                "expert_capacity": 1.0,
            }
    
    # 3. 创建模型（不加载权重，只构建结构）
    model = WhisperForConditionalGeneration._from_config(config)
    
    # 4. 重建MoE结构（关键步骤）
    model = replace_ffn_with_deepspeed_moe(model, config, moe_config)
    
    # 5. 查找并加载safetensors权重文件
    safetensors_files = glob.glob(os.path.join(model_path, "model*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"在{model_path}中找不到safetensors文件")
    
    # 6. 加载所有权重
    state_dict = {}
    for file in sorted(safetensors_files):
        print(f"📥 加载权重: {os.path.basename(file)}")
        state_dict.update(load_file(file))
    
    # 7. 应用到模型（strict=False允许部分不匹配）
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    # 8. 打印加载状态
    if missing_keys:
        print(f"⚠️  缺失权重（未加载）: {missing_keys[:5]}...")  # 只显示前5个
    if unexpected_keys:
        print(f"⚠️  意外权重（模型中不存在）: {unexpected_keys[:5]}...")
    
    print(f"✅ 模型加载完成，总权重数: {len(state_dict)}")
    model = model.to(torch.bfloat16)
    
    return model

def init_deepspeed_environment(args):
    """
    初始化DeepSpeed环境（仅初始化，不加载优化器）
    """
    if args.local_rank != -1:
        # DeepSpeed已启动，初始化通信后端
        if not torch.distributed.is_initialized():
            deepspeed.init_distributed(
                dist_backend="nccl" if torch.cuda.is_available() else "gloo"
            )
        
        # 获取分布式信息
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        
        # 设置当前GPU
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        print(f"🖥️  Rank {rank}/{world_size} - GPU:{args.local_rank}")
    else:
        # 单卡模式
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Single GPU mode - Device: {device}")
    
    return device

def load_model_with_deepspeed(args):
    """
    加载模型并适配DeepSpeed环境
    """
    # 加载模型（确保在正确的device上）
    model = load_whisper_moe_model(args.model_path)
    
    # 移动到当前GPU
    if args.local_rank != -1:
        model = model.to(f"cuda:{args.local_rank}")
    else:
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # 设置为评估模式
    model.eval()
    
    return model

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# 评估函式，现在只负责对一个已经载入的数据集进行评估
def run_evaluation(model, processor, eval_dataset, args):
    """
    对给定的单个数据集进行评估并返回 WER。
    """
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=data_collator)

    data = pd.DataFrame(columns=["hypothesis", "reference"])
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad(), torch.cuda.amp.autocast():
            # ... (model.generate 和解码的逻辑完全不变) ...
            generated_tokens = (
                model.generate(
                    input_features=batch["input_features"].to(args.device),
                    max_new_tokens=args.max_new_tokens,
                ).cpu().numpy()
            )
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
            decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
            data = pd.concat([data, pd.DataFrame({"hypothesis": decoded_preds, "reference": decoded_labels})], ignore_index=True)
    
    # 文本清洗
    normalizer = BasicTextNormalizer()
    data1 = pd.DataFrame(columns=["hypothesis_clean", "reference_clean"])
    for i in range(len(data)):
        hyp = normalizer(data["hypothesis"][i])
        ref = normalizer(data["reference"][i])
        data1 = pd.concat([data1, pd.DataFrame({"hypothesis_clean": [hyp], "reference_clean": [ref]})], ignore_index=True)
    
    # 计算 WER
    wer = jiwer.wer(list(data1["reference_clean"]), list(data1["hypothesis_clean"]))
    return wer * 100

# 主程式部分
if __name__ == "__main__":
    # 1. 定义你的语言映射字典
    language_mapping = {
        "id_id": "indonesian",
        "ms_my": "malay",
        "fil_ph": "tagalog",
        "en_us": "english",
        "zh_cn": "chinese",
        "th_th": "thai",
        "vi_vn": "vietnamese",
        "jv_id": "javanese",
        "mi_nz": "maori",
        "zh-CN": "chinese",
        "de_de": "german",
        "es_es": "spanish",
        "fr_fr": "french",
        "it_it": "italian",
        "ja_jp": "japanese",
        "ko_kr": "korean",
        "pt_pt": "portuguese",
        "km_kh": "khmer",
        "my_mm": "burmese",
        "lo_la": "lao",
        "hy_am": "armenian",
        "lt_lt": "lithuanian",
        "cy_gb": "welsh",
        "fa_ir": "persian",
        "he_il": "hebrew",
        "is_is": "icelandic",
    }

    # 将所有要测试的语言设置放在这里
    all_datasets_settings = [
        ["fleurs", {"language_abbr": "my_mm"}],
        ["fleurs", {"language_abbr": "km_kh"}],
        ["fleurs", {"language_abbr": "lo_la"}],
        ["fleurs", {"language_abbr": "jv_id"}],
        ["fleurs", {"language_abbr": "mi_nz"}],
        
        ["fleurs", {"language_abbr": "ms_my"}],
        ["fleurs", {"language_abbr": "id_id"}],
        ["fleurs", {"language_abbr": "fil_ph"}],
        ["fleurs", {"language_abbr": "th_th"}],
        ["fleurs", {"language_abbr": "vi_vn"}],
    ]
    
    all_lang_results = []
    parser = argparse.ArgumentParser()

    # 模型参数设置
    parser.add_argument("--model_path", default="", type=str)
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--language", default="german")  # 设置语言
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_new_tokens", default=128, type=int)
    parser.add_argument("--metric", default="wer")
    parser.add_argument("--device", default="cuda:0")
    # 数据集设置
    parser.add_argument("--num_test_samples", default=1000, type=int)
    parser.add_argument("--max_input_length", default=30.0, type=float)
    parser.add_argument("--test_only", default=True, action="store_true")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--num_proc", default=1, type=int)
    parser.add_argument("--save_path", default="eval_results/default", help="保存评估结果的目录")
    parser.add_argument("--local_rank", type=int, default=-1, 
                       help="自动注入的本地rank")
    parser.add_argument("--deepspeed", type=str, default=None,
                       help="DeepSpeed配置文件路径（评估时不需要）")

    args = parser.parse_args()
    
    # 💡 关键点：初始化DeepSpeed通信环境
    args.device = init_deepspeed_environment(args)  # 添加这行
    
    model_path = args.model_path
    processor_path = model_path
    print(f"Settings: {args}")
    os.makedirs(args.save_path, exist_ok=True)

    output_csv  = os.path.join(args.save_path, "eval_wer_summary.csv")
    output_json = os.path.join(args.save_path, "eval_wer_summary.json")
    # 💡 关键：先把模型和处理器载入一次，避免在循环中重复载入
    print(f"Loading processor from: {processor_path}")
    processor = WhisperProcessor.from_pretrained(processor_path) # 从原始路径加载处理器

    print(f"Loading model from: {model_path}") # 或者直接用 model_path
    model = load_whisper_moe_model(model_path) # 从checkpoint加载模型

    model.to(args.device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型总参数量: {total_params / 1e9:.2f}B")
    print(f"📊 可训练参数量: {trainable_params / 1e9:.2f}B")

    # 循环处理每个语言
    for single_dataset_setting in all_datasets_settings:
        lang_abbr = single_dataset_setting[1]["language_abbr"]
        # 2. 从映射字典中获取正确的Whisper语言名称
        whisper_lang = language_mapping.get(lang_abbr)
        # 安全检查，如果字典里没有这个语言就跳过
        if whisper_lang is None:
            print(f"警告：在映射字典中找不到 {lang_abbr} 的对应语言，跳过此语言。")
            continue
        print(f"--- 开始评估语言: {lang_abbr} ---")
        
        # 每次只载入一个语言的数据集
        ds = load_process_datasets(
            [single_dataset_setting],  # 注意这里要用列表包起来
            processor,
            max_input_length=args.max_input_length,
            num_test_samples=args.num_test_samples,
            test_only=args.test_only,
            streaming=args.streaming,
            num_proc=args.num_proc,
            dataset_root='/mnt/lv2/FLEURS',
            json_output_dir="/mnt/lv2/FLEURS/json"
        )

        # 设定当前评估的语言
        # 这一步很重要，告诉模型现在要识别哪个语言
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=whisper_lang, task=args.task)

        # 对当前语言的数据集进行评估
        wer = run_evaluation(model, processor, ds["test"], args)

        # 打印当前语言的结果
        print(f"✅ 语言 {lang_abbr} 的 WER: {wer:.2f} %")
        print("--- 评估结束 ---\n")
        
        all_lang_results.append({"language": lang_abbr,
                                 "wer": round(float(wer), 2),
                                 "timestamp": str(datetime.datetime.now())})
        pd.DataFrame(all_lang_results).to_csv(output_csv, index=False)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_lang_results, f, ensure_ascii=False, indent=2) 