import os
os.environ['TMPDIR'] = '/mnt/lv3/aiyuchen/temp'
import gc
import random
import librosa
import numpy as np
from load_each import load_common_voice, load_fleurs
from functools import partial

from datasets import (load_dataset, 
                      concatenate_datasets,
                      IterableDatasetDict,
                      DatasetDict, 
                      Dataset,
                      Audio,
)

from audiomentations import (
    AddGaussianNoise,
    Compose,
    Gain,
    OneOf,
    PitchShift,
    TimeStretch,
)

from torchlibrosa.augmentation import SpecAugmentation

def load_filepaths_and_text(filename, split=","):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text

def freq_mask(spec, F=10):
    num_mels = spec.shape[0]
    f = np.random.randint(0, F)  # 掩码长度
    f0 = np.random.randint(0, num_mels - f) # 起点
    spec[f0:f0+f, :] = 0
    return spec

def time_mask(spec, T=50):
    num_frames = spec.shape[1]
    t = np.random.randint(0, T)  # 掩码长度
    t0 = np.random.randint(0, num_frames - t) # 起点
    spec[:, t0:t0+t] = 0
    return spec

def spec_augment(features):
    # 应用增强
    features = freq_mask(features)
    features = time_mask(features)
    return features

def prepare_dataset(batch, augment_data=2, processor=None):
    # load and (possibly) resample audio data to 16kHz
    # print("batch keys:", batch["audio"].keys())
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    if augment_data == 2:
        input_features = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        augmented_features = spec_augment(input_features)
        batch["input_features"] = augmented_features
    else:
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    
    # compute input length of audio sample in seconds
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    # print(batch["language"])
    
    language_name = {
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
    
    # 正确设置语言标签
    processor.tokenizer.set_prefix_tokens(
        language=language_name[batch["language"]], task="transcribe")

    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    # batch["language"] = batch["language"]
    return batch
    
def augment_dataset(batch):
    # load and (possibly) resample audio data to 16kHz
    sample = batch["audio"]
    
    # 数据增强操作
    augmentation = Compose(
            [
                TimeStretch(min_rate=0.9, max_rate=1.1, p=0.2,
                            leave_length_unchanged=False),
                Gain(-6, 6, p=0.1),
                PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
                OneOf(
                    [
                        AddGaussianNoise(min_amplitude=0.005,
                                            max_amplitude=0.015, p=1.0),
                    ],
                    p=0.2,
                ),
            ]
        )
    
    augmented_waveform = augmentation(
        sample["array"], sample_rate=sample["sampling_rate"])

    batch["audio"]["array"] = augmented_waveform

    return batch

def augment_dataset2(batch):
    # 提取音频
    audio_array = batch["audio"]["array"]
    sample_rate = batch["audio"]["sampling_rate"]
    
    # 计算音频梅尔频谱
    mel_spec = librosa.feature.melspectrogram(
        y=audio_array,
        sr=sample_rate,
        n_fft=400,
        hop_length=160,
        win_length=400,
        n_mels=80,
        fmin=0,
        fmax=8000
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 定义数据增强
    def time_warp(spec, W=5):
        """
        对 log-mel 频谱图进行时间轴的扭曲（仿射拉伸）
        参数：
            spec: 形状为 (n_mels, time_frames)
            W: 扭曲窗口大小
        返回：
            扭曲后的 spec
        """
        num_frames = spec.shape[1]
        if num_frames <= W * 2:
            return spec

        center = np.random.randint(W, num_frames - W)
        warped = np.random.randint(center - W, center + W)

        # 构建插值映射：输入是每一行（mel bin）的时间帧
        def warp_line(line):
            return np.interp(
                np.arange(num_frames),
                [0, center, num_frames - 1],
                [0, warped, num_frames - 1],
                left=0,
                right=0
            )

        # 应用到所有 Mel bin
        warped_spec = np.zeros_like(spec)
        for m in range(spec.shape[0]):
            warped_spec[m, :] = np.interp(
                np.arange(num_frames),
                warp_line(np.arange(num_frames)),
                spec[m, :],
                left=0,
                right=0
            )
        return warped_spec
    
    # 应用增强
    # log_mel_spec = time_warp(log_mel_spec)
    log_mel_spec = freq_mask(log_mel_spec)
    log_mel_spec = time_mask(log_mel_spec)
    
    # 转换为音频
    mel_spec = librosa.db_to_power(log_mel_spec)
    augmented_waveform = librosa.feature.inverse.mel_to_audio(
        mel_spec,
        sr=sample_rate,
        n_fft=400,
        hop_length=160,
        win_length=400,
    )
        
    batch["audio"]["array"] = augmented_waveform

    return batch

def load_process_datasets(
    datasets_settings, 
    processor, 
    dataset_root,
    json_output_dir,
    streaming = True, 
    cache_dir = None, 
    test_only = False, 
    num_test_samples = 1000,
    sampling_rate = 16000, 
    max_input_length = 30.0, 
    num_proc = 1, 
    buffer_size = 500, 
    seed = 42, 
    language_abbr = "id_id", 
    replay_ratio = 0.0, 
    augment_data = 0
    ):
    
    def is_audio_in_length_range(length):
        return length < max_input_length
    
    # sampling_rate = processor.feature_extractor.sampling_rate
    all_datasets = {"train": [], "test": []}

    if not cache_dir:
        cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/datasets"))

    if streaming:
        ds = IterableDatasetDict()
    else:
        ds = DatasetDict()
        
    ds.clear()

    train_list, test_list = [], []
    for name, kwargs in datasets_settings:
        print(f"Loading dataset: {name} with args {kwargs}")
        
        ds_tmp = None
        if name == "fleurs":
            # kwargs should contain 'language_abbr'
            if "language_abbr" not in kwargs:
                raise ValueError("language_abbr must be specified for fleurs dataset")
            ds_tmp = load_fleurs(
                dataset_root, 
                sampling_rate=sampling_rate,
                streaming=streaming, 
                cache_dir=cache_dir, 
                test_only=test_only, 
                replay_ratio=replay_ratio,
                json_output_dir=json_output_dir,
                **kwargs
                )
            print(f"fleurs-{kwargs}: ", next(iter(ds_tmp["test"])))
        elif name == "common_voice":
            if "language_abbr" not in kwargs:
                raise ValueError("language_abbr must be specified for common_voice_local dataset")
            ds_tmp = load_common_voice(
                dataset_root, 
                sampling_rate=sampling_rate,
                streaming=streaming, 
                cache_dir=cache_dir, 
                test_only=test_only, 
                replay_ratio=replay_ratio,
                json_output_dir=json_output_dir,
                **kwargs
                )
            print(f"common_voice-{kwargs}: ", next(iter(ds_tmp["test"])))

        if ds_tmp is not None:
            test_list.append(ds_tmp["test"])
            if not test_only:
                train_list.append(ds_tmp["train"])
                # augment training data
                if augment_data == 1 or augment_data == 2:
                    augmented_raw_training_dataset = ds_tmp["train"].map(
                        augment_dataset, num_proc=1, desc="augment train dataset")  # num_proc=16
                    train_list.append(augmented_raw_training_dataset)
        
    
    # ds是多个ds_tmp拼接而成
    ds["test"] = concatenate_datasets(test_list)
    if not test_only:
        ds["train"] = concatenate_datasets(train_list)

    prepare_dataset_with_augment = partial(
        prepare_dataset, 
        augment_data=augment_data,
        processor=processor,
    )
    
    # 对数据进行处理
    if streaming:
        remove_columns = [col for col in list(next(iter(ds.values())).features) if col not in ["language"]]
        ds = ds.map(prepare_dataset_with_augment,
                    remove_columns=remove_columns,
                    ).with_format("torch")
        print("ds: ", ds)
        ds = ds.filter(
            is_audio_in_length_range, input_columns=["input_length"])
        print("ds_filtered: ", ds)
        ds = ds.shuffle(seed, buffer_size=buffer_size)
        ds["test"] = ds["test"].take(num_test_samples)
    else:
        ds = ds.shuffle(seed)
        num_test_samples = min(num_test_samples, ds["test"].num_rows)
        ds["test"] = ds["test"].select(range(num_test_samples))
        remove_columns = [col for col in list(next(iter(ds.values())).features) if col not in ["language"]]
        ds = ds.map(
            prepare_dataset_with_augment,
            remove_columns=remove_columns,
            num_proc=num_proc,
            ).with_format("torch")
        
        ds = ds.filter(
            is_audio_in_length_range, 
            input_columns=["input_length"]
            )
        
    # print("训练集大小: ", len(ds["train"]))
    print("ds: ", ds)
    return ds

if __name__ == "__main__":
    from transformers import WhisperProcessor, WhisperTokenizer

    # Model setups
    model_name_or_path = "Oblivion208/whisper-tiny-cantonese"
    task = "transcribe"
    language = "zh"
    # Dataset setups
    datasets_settings = [
        ["mdcc", {}],
        ["common_voice", {"language_abbr": "zh-HK"}],
    ]
    max_input_length = 30.0
    num_test_samples = 1000

    tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, task=task)
    processor = WhisperProcessor.from_pretrained(model_name_or_path, task=task)
    ds = load_process_datasets(
        datasets_settings,
        processor,
        max_input_length=max_input_length,
        num_test_samples=num_test_samples,
        test_only=True,
        streaming=False,
        num_proc=4,
    )
    print(ds)
    print("test", ds["test"][:10]["input_length"])
    # print("train", ds["train"][:10]["input_length"])
    # print("test sample: ", next(iter(ds["test"])))
