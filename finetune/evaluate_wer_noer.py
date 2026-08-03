import os, json, csv, datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import argparse
import gc
import jiwer
import pandas as pd
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from whisper_normalizer.basic import BasicTextNormalizer
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from tqdm import tqdm
from load_datasets import load_process_datasets
import zhconv
from datasets import load_from_disk

# 2. 在所有 transformers 代码之前，设置这个环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        with torch.no_grad():
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
    # 模型权重路径，指向你找到的完好的checkpoint
    model_path = "/mnt/lv2/FLEURS/whisper-finetune-results/ALL/medium_noer"
    
    # 处理器路径，指向你训练时使用的、文件完整的原始基础模型
    # 从你的训练脚本中可以看到这个路径
    processor_path = "/mnt/lv2/FLEURS/whisper-medium" 

    # 1. 定义你的语言映射字典
    language_mapping = {
        "ms_my": "malay", "id_id": "indonesian", "fil_ph": "tagalog",
        "jv_id": "javanese", "mi_nz": "maori", "th_th": "thai",
        "vi_vn": "vietnamese", "en_us": "english", "fr_fr": "french",
    }

    # 将所有要测试的语言设置放在这里
    all_datasets_settings = [
        ["fleurs", {"language_abbr": "ms_my"}],
        ["fleurs", {"language_abbr": "id_id"}],
        ["fleurs", {"language_abbr": "fil_ph"}],
        ["fleurs", {"language_abbr": "jv_id"}],
        ["fleurs", {"language_abbr": "mi_nz"}],
        ["fleurs", {"language_abbr": "th_th"}],
        ["fleurs", {"language_abbr": "vi_vn"}],
        ["fleurs", {"language_abbr": "en_us"}],
        ["fleurs", {"language_abbr": "fr_fr"}],
    ]
    
    all_lang_results = []
    parser = argparse.ArgumentParser()

    # 模型参数设置
    parser.add_argument("--model_name_or_path", default=model_path, type=str)
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--language", default="german")  # 设置语言
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_new_tokens", default=255, type=int)
    parser.add_argument("--metric", default="wer")
    parser.add_argument("--device", default="cuda:0")
    # 数据集设置
    parser.add_argument("--num_test_samples", default=1000, type=int)
    parser.add_argument("--max_input_length", default=30.0, type=float)
    parser.add_argument("--test_only", default=True, action="store_true")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--num_proc", default=1, type=int)
    parser.add_argument("--save_path", default="eval_results/medium_noer", help="保存评估结果的目录")

    args = parser.parse_args()
    print(f"Settings: {args}")
    os.makedirs(args.save_path, exist_ok=True)

    output_csv  = os.path.join(args.save_path, "eval_wer_summary.csv")
    output_json = os.path.join(args.save_path, "eval_wer_summary.json")
    # 💡 关键：先把模型和处理器载入一次，避免在循环中重复载入
    print(f"Loading processor from: {processor_path}")
    processor = WhisperProcessor.from_pretrained(processor_path) # 从原始路径加载处理器

    print(f"Loading model from: {args.model_name_or_path}") # 或者直接用 model_path
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path) # 从checkpoint加载模型

    model.to(args.device)
    model.eval()

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