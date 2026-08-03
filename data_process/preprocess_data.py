import argparse
from datasets import concatenate_datasets, DatasetDict
from load_datasets import load_process_datasets
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import os
os.environ['TMPDIR'] = '/mnt/lv3/aiyuchen/temp'

def load_model_and_processor(model_name_or_path: str, language: str, task: str):
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
    processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)
    return model, processor

def main():
    parser = argparse.ArgumentParser(description="Preprocess and save datasets")
    # 你只需要关心数据相关的参数和存储路径
    parser.add_argument("--dataset_root", default="/mnt/lv2/FLEURS")
    parser.add_argument("--json_output_dir", default="/mnt/lv2/FLEURS/json")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max_input_length", default=30, type=float)
    parser.add_argument("--num_test_samples", default=1000, type=int)
    parser.add_argument("--num_proc", default=1, type=int)
    parser.add_argument("--replay_ratio", default=0, type=float)
    # ！！！最关键的参数：指定处理好的数据要存到哪里！！！
    parser.add_argument("--output_dir", default="/mnt/lv2/FLEURS/cache")

    args = parser.parse_args()

    # 只需要 processor 来处理数据，不需要加载完整模型
    model_name_or_path = "/mnt/lv2/FLEURS/whisper-medium"
    _, processor = load_model_and_processor(model_name_or_path, language="id", task="transcribe")

    # --- 你的数据集设定可以保持不变 ---
    datasets_settings = [
        # ["fleurs", {"language_abbr": "my_mm"}],
        # ["fleurs", {"language_abbr": "km_kh"}],
        # ["fleurs", {"language_abbr": "lo_la"}],
        # ["fleurs", {"language_abbr": "jv_id"}],
        ["fleurs", {"language_abbr": "mi_nz"}],
        ["fleurs", {"language_abbr": "th_th"}],
        
        ["fleurs", {"language_abbr": "is_is"}],
        ["fleurs", {"language_abbr": "cy_gb"}],
        
        ["fleurs", {"language_abbr": "hy_am"}],
        ["fleurs", {"language_abbr": "lt_lt"}],
        
        ["fleurs", {"language_abbr": "he_il"}],
        ["fleurs", {"language_abbr": "fa_ir"}],
    ]
    
    """
    ER_datasets = [
        ["fleurs", {"language_abbr": "th_th"}],
        ["fleurs", {"language_abbr": "vi_vn"}],
        ["fleurs", {"language_abbr": "en_us"}],
        ["fleurs", {"language_abbr": "fr_fr"}],
        ["fleurs", {"language_abbr": "zh_cn"}],
    ]
    """

    # --- 执行所有数据处理步骤 ---
    # print("📜 开始数据预处理...")
    
    print("📜 开始处理主数据集 (datasets_settings)...")
    ds = load_process_datasets(
        datasets_settings,
        processor,
        max_input_length=args.max_input_length,
        json_output_dir=args.json_output_dir,
        dataset_root=args.dataset_root,
        num_test_samples=args.num_test_samples,
        streaming=args.streaming,
        num_proc=args.num_proc,
        augment_data = 2,
    )
    main_data_output_path = os.path.join(args.output_dir, "main_data_med_new")
    print(f"✅ 正在保存主数据集到 {main_data_output_path}")
    ds.save_to_disk(main_data_output_path)
    

if __name__ == "__main__":
    main()