import librosa
import numpy as np
from datasets import load_dataset, concatenate_datasets, Audio
from transformers import WhisperProcessor


# =============================
# SpecAugment
# =============================
def spec_augment(features, num_mask=2, freq_masking=0.15, time_masking=0.15):
    augmented = features.copy()
    num_mel = augmented.shape[0]
    num_time = augmented.shape[1]

    for _ in range(num_mask):
        freq_width = int(num_mel * freq_masking)
        f0 = np.random.randint(0, num_mel - freq_width)
        augmented[f0:f0 + freq_width, :] = 0

        time_width = int(num_time * time_masking)
        t0 = np.random.randint(0, num_time - time_width)
        augmented[:, t0:t0 + time_width] = 0

    return augmented


# =============================
# Whisper语言映射
# =============================
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
}


# =============================
# 读取audio（兼容所有格式）
# =============================
def load_audio(audio):

    if isinstance(audio, dict):

        if "array" in audio:
            return audio["array"], audio["sampling_rate"]

        if "path" in audio:
            audio_array, sr = librosa.load(audio["path"], sr=None)
            return audio_array, sr

    if isinstance(audio, np.ndarray):
        return audio, 16000

    raise ValueError(f"Unsupported audio format: {type(audio)}")


# =============================
# 数据预处理
# =============================
def prepare_dataset(batch, processor=None, augment_data=False):

    audio_array, sampling_rate = load_audio(batch["audio"])

    # Whisper feature
    input_features = processor.feature_extractor(
        audio_array,
        sampling_rate=sampling_rate
    ).input_features[0]

    if augment_data:
        input_features = spec_augment(input_features)

    batch["input_features"] = input_features

    batch["input_length"] = len(audio_array) / sampling_rate

    # language token
    processor.tokenizer.set_prefix_tokens(
        language=language_name[batch["language"]],
        task="transcribe"
    )

    batch["labels"] = processor.tokenizer(
        batch["sentence"]
    ).input_ids

    return batch


# =============================
# 加载dataset
# =============================
def load_process_datasets(
        dataset_name,
        processor,
        split="test",
        sampling_rate=16000,
        num_workers=4):

    ds = load_dataset(dataset_name, split=split)

    # 强制decode audio
    ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))

    remove_columns = [
        col for col in ds.column_names
        if col not in ["audio", "sentence", "language"]
    ]

    ds = ds.map(
        lambda x: prepare_dataset(x, processor),
        remove_columns=remove_columns,
        num_proc=num_workers
    )

    return ds