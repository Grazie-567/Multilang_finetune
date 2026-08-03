import os
import gc
from datasets import Dataset, DatasetDict, IterableDatasetDict, load_dataset, concatenate_datasets, Audio

def load_filepaths_and_text(filename, split=","):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text

def create_dataset(
    dataset_dir, 
    ds_keys, 
    audio_paths, 
    transcription_texts,
    sampling_rate, 
    streaming, 
    cache_dir, 
    use_valid_to_train, 
    test_only, 
    languages, 
    json_output_dir
):
    if streaming:
        ds = IterableDatasetDict()
    else:
        ds = DatasetDict()

    for key in ds_keys:
        dataset_dict = {
            "audio": audio_paths[key], "sentence": transcription_texts[key], "language":languages[key]}
        ds_tmp = Dataset.from_dict(dataset_dict)

        os.makedirs(json_output_dir, exist_ok=True)
        
        # 2. 构建新的、唯一的文件路径，避免不同数据集间冲突
        lang_abbr = languages[key][0] if languages[key] else "unknown"
        json_path = os.path.join(json_output_dir, f"{lang_abbr}_{key}.json")
        
        # 3. 写入并读取
        if not os.path.exists(json_path):
            print(f"Writing intermediate json to: {json_path}")
            ds_tmp.to_json(json_path, index=False)
        
        # 读取json文件中信息
        ds[key] = load_dataset("json", data_files=json_path, split='train',
                               features=ds_tmp.features,
                               streaming=streaming,
                               cache_dir=cache_dir,
                               )

    del ds_tmp
    gc.collect()

    if use_valid_to_train and not test_only:
        ds["train"] = concatenate_datasets([ds["train"], ds["dev"]])

    ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))
    return ds

def load_common_voice(
    dataset_root,
    language_abbr="ms_my",
    sampling_rate=16000,
    streaming=True,
    cache_dir="/mnt/lv3/chenkaizhe/.cache/huggingface/datasets",
    use_valid_to_train=True,
    test_only=False,
    replay_ratio=0.0,
    json_output_dir="/mnt/lv3/renziang/json_fleurs"
    ):
    
    # dataset_dir = dataset_root +  "CommonVoice/" + language_abbr + "/"
    dataset_dir = os.path.join(dataset_root, "CommonVoice", language_abbr)
    # print("loading dataset dir:", dataset_dir)
    
    if test_only:
        ds_keys = ["test"]
    else:
        ds_keys = ["train", "dev", "test"]
    
    audio_paths, transcription_texts, languages = {}, {}, {}
    for key in ds_keys:
        # filelist = dataset_dir + f"{key}.tsv"
        filelist = os.path.join(dataset_dir, f"{key}.tsv")
        # print(filelist)
        filepaths_and_text = load_filepaths_and_text(filelist,split='\t')
        filepaths_and_text[0].append("transcription")
        
        # 存入相关信息
        audio_paths[key], transcription_texts[key], languages[key] = [], [], []
        
        # 经验回放，抽取一部分数据集
        if replay_ratio > 0:
            indices = range(len(filepaths_and_text))
            sample = random.sample(indices, round(len(filepaths_and_text) * replay_ratio))
            # print("取样大小: ", len(sample),"/", len(filepaths_and_text),language_abbr)
            # print(sample)
            # print(filepaths_and_text)
            for k in sample:
                if k == 0: 
                    continue
                # audio_path = dataset_dir + "clips/" + filepaths_and_text[k][1]
                audio_path = os.path.join(dataset_dir, "clips", filepaths_and_text[k][1])
                audio_paths[key].append(audio_path)
                if language_abbr == 'zh-CN':
                    transcript = filepaths_and_text[k][2].replace("«","").replace("»","")
                else:
                    transcript = filepaths_and_text[k][3].replace("«","").replace("»","")
                transcription_texts[key].append(transcript)
                # 加入语种信息
                languages[key].append(language_abbr)
        else:
            # 如数据集过多可以downsample
            for i in range(1, len(filepaths_and_text)):
                audio_path = os.path.join(dataset_dir, "clips", filepaths_and_text[i][1])
                audio_paths[key].append(audio_path)
                # print(filepaths_and_text)
                if language_abbr == 'zh-CN':
                    transcript = filepaths_and_text[i][2].replace("«","").replace("»","")
                else:
                    transcript = filepaths_and_text[i][3].replace("«","").replace("»","")
                transcription_texts[key].append(transcript)
                # 加入语种信息
                languages[key].append(language_abbr)
             
    ds = create_dataset(
        dataset_dir = dataset_dir, 
        ds_keys = ds_keys, 
        audio_paths = audio_paths, 
        transcription_texts = transcription_texts,
        sampling_rate = sampling_rate, 
        streaming = streaming, 
        cache_dir = cache_dir,
        use_valid_to_train = use_valid_to_train, 
        test_only = test_only, 
        languages = languages, 
        json_output_dir = json_output_dir
    )
        
    return ds

def load_fleurs(
    dataset_root, 
    language_abbr="ms_my", 
    sampling_rate=16000, 
    streaming=True, 
    cache_dir="/mnt/lv3/chenkaizhe/.cache/huggingface/datasets", 
    use_valid_to_train=True, 
    test_only=False, 
    replay_ratio=0.0, 
    json_output_dir="/mnt/lv3/renziang/json_fleurs"
    ):
    
    # dataset_dir = dataset_root + language_abbr + "/"
    #import pdb; pdb.set_trace()
    dataset_dir = os.path.join(dataset_root, language_abbr)
    # print("loading dataset dir:", dataset_dir)
    
    if test_only:
        ds_keys = ["test"]
    else:
        ds_keys = ["train", "dev", "test"]
    
    audio_paths, transcription_texts, languages = {}, {}, {}
    for key in ds_keys:
        # filelist = dataset_dir + f"{key}.tsv"
        filelist = os.path.join(dataset_dir, f"{key}.tsv")
        # print(filelist)
        filepaths_and_text = load_filepaths_and_text(filelist,split='\t')
        filepaths_and_text[0].append("transcription")
        # print(filepaths_and_text[0][1])
        
        # 存入相关信息
        audio_paths[key], transcription_texts[key], languages[key] = [], [], []
        # 经验回放，抽取一部分数据集
        if replay_ratio > 0:
            indices = range(len(filepaths_and_text))
            sample = random.sample(indices, round(len(filepaths_and_text) * replay_ratio))
            # print("取样大小: ", len(sample),"/", len(filepaths_and_text),language_abbr)
            # print(sample)
            for k in sample:
                # audio_path = dataset_dir + "audio/" + key + "/" + filepaths_and_text[k][1]
                audio_path = os.path.join(dataset_dir, "audio", key, filepaths_and_text[k][1])
                audio_paths[key].append(audio_path)
                transcript = filepaths_and_text[k][3].replace("«","").replace("»","")
                transcription_texts[key].append(transcript)
                # 加入语种信息
                languages[key].append(language_abbr)
        else:
            for i in range(1, len(filepaths_and_text)):
                # audio_path = dataset_dir + "audio/" + key + "/" + filepaths_and_text[i][1]
                audio_path = os.path.join(dataset_dir, "audio", key, filepaths_and_text[i][1])
                audio_paths[key].append(audio_path)
                transcript = filepaths_and_text[i][3].replace("«","").replace("»","")
                transcription_texts[key].append(transcript)
                # 加入语种信息
                languages[key].append(language_abbr)
    
    # 创建数据集         
    ds = create_dataset(
        dataset_dir = dataset_dir, 
        ds_keys = ds_keys, 
        audio_paths = audio_paths, 
        transcription_texts = transcription_texts,
        sampling_rate = sampling_rate, 
        streaming = streaming, 
        cache_dir = cache_dir,
        use_valid_to_train = use_valid_to_train, 
        test_only = test_only, 
        languages = languages, 
        json_output_dir = json_output_dir
    )
    print(f"Keys available in ds: {ds.keys()}") # 添加这一行来调试
    # print("ds_train: ", len(ds["train"]))
    return ds  