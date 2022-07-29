import argparse
import os
from glob import glob
from argparse import RawTextHelpFormatter

import torch
from tqdm import tqdm

from TTS.tts.configs.shared_configs import BaseDatasetConfig

from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.managers import save_file
from TTS.tts.utils.speakers import SpeakerManager

parser = argparse.ArgumentParser(
    description="""Compute embedding vectors for each wav file in a dataset.\n\n"""
                """
                Example runs:
                python TTS/bin/compute_embeddings.py speaker_encoder_model.pth speaker_encoder_config.json  dataset_config.json
                """,
    formatter_class=RawTextHelpFormatter,
)

use_cuda = torch.cuda.is_available()

CONFIG_SE_PATH = "config_se.json"
CHECKPOINT_SE_PATH = "SE_checkpoint.pth.tar"
output_path = ''

dataset_path = r"C:\Users\PiaoYang\Desktop\aidatatang_200zh"
dataset_config = [
    BaseDatasetConfig(meta_file_train='aidatatang_200_zh_transcript.csv', path=dataset_path)
]


# load training samples
def formatter(root_path, meta_file=None, ignored_speakers=None):
    filename_text_dict = {}
    with open(os.path.join(root_path, meta_file), 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            filename, text = line.split('|')
            filename_text_dict[filename] = text

    items = []
    for wav_file_path in glob(f"{os.path.join(root_path, 'corpus/train')}/**/*.wav", recursive=True):
        wav_filename = os.path.basename(wav_file_path).split('.wav')[0]
        items.append({'audio_file': wav_file_path,
                      'text': filename_text_dict[wav_filename],
                      'speaker_name': wav_filename[5:10]})

    return items


train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    formatter=formatter
)

meta_data_train = train_samples
meta_data_eval = eval_samples

if meta_data_eval is None:
    wav_files = meta_data_train
else:
    wav_files = meta_data_train + meta_data_eval

encoder_manager = SpeakerManager(
    encoder_model_path=CHECKPOINT_SE_PATH,
    encoder_config_path=CONFIG_SE_PATH,
    use_cuda=use_cuda,
)

class_name_key = 'speaker_name'

# compute speaker embeddings
speaker_maping = {}
for idx, wav_file in enumerate(tqdm(wav_files)):
    if isinstance(wav_file, dict):
        class_name = wav_file[class_name_key]
        wav_file = wav_file["audio_file"]
    else:
        class_name = None
        raise Exception('Invalid Speaker name')

    wav_file_name = os.path.basename(wav_file)
    # extract the embedding
    embedd = encoder_manager.compute_embedding_from_clip(wav_file)

    # create speaker_mapping if target dataset is defined
    speaker_mapping[wav_file_name] = {}
    speaker_mapping[wav_file_name]["name"] = class_name
    speaker_mapping[wav_file_name]["embedding"] = embedd

if speaker_mapping:
    # save speaker_mapping if target dataset is defined
    if os.path.isdir(output_path):
        mapping_file_path = os.path.join(output_path, "speakers.pth")
    else:
        mapping_file_path = output_path

    if os.path.dirname(mapping_file_path) != "":
        os.makedirs(os.path.dirname(mapping_file_path), exist_ok=True)

    save_file(speaker_mapping, mapping_file_path)
    print("Speaker embeddings saved at:", mapping_file_path)
