import os
from glob import glob
import torch

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

from TTS.tts.utils.text.characters import _phonemes

output_path = os.path.dirname(os.path.abspath(__file__))

dataset_path = r"C:\Users\PiaoYang\Desktop\aidatatang_200zh"
folder_filter = '**'
dataset_config = [
    BaseDatasetConfig(meta_file_train='aidatatang_200_zh_transcript.csv', path=dataset_path)
]

audio_config = VitsAudioConfig(
    sample_rate=16000,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0,
    mel_fmax=None,
)

d_vector_path = r'speakers_sub.pth'
vitsArgs = VitsArgs(
    use_language_embedding=False,
    embedded_language_dim=4,

    use_speaker_embedding=False,
    # speaker_embedding_channels=512,

    use_d_vector_file=True,
    d_vector_file=d_vector_path,
    d_vector_dim=512,
)

config = VitsConfig(
    run_name="vits_200zh",

    # Model Architecture
    model_args=vitsArgs,

    # Dataset
    datasets=dataset_config,

    # Audio
    audio=audio_config,
    min_audio_len=32 * 256 * 4,
    max_audio_len=160000,

    # Trainer
    # Use default batch size and dataloader worker
    output_path=output_path,

    batch_size=32,
    eval_batch_size=16,
    precompute_num_workers=8,
    num_loader_workers=8,
    num_eval_loader_workers=4,

    scheduler_after_epoch=True,
    dur_loss_alpha=2,
    feat_loss_alpha=2,
    kl_loss_alpha=2,

    epochs=1000,
    print_step=100,

    run_eval=True,
    print_eval=True,
    run_eval_steps=500,

    save_step=200,
    save_n_checkpoints=3,
    save_best_after=0,

    test_sentences=[['我最爱我的宝了', 'G0991'], ['大猪头就是大猪头', 'G0991']],

    # Tokenizer
    text_cleaner="multilingual_cleaners",
    use_phonemes=True,
    compute_input_seq_cache=True,
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    phoneme_language="zh-cn",
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters="!¡'(),-.:;¿?abcdefghijklmnopqrstuvwxyzµßàáâäåæçèéêëìíîïñòóôöùúûüąćęłńœśşźżƒабвгдежзийклмнопрстуфхцчшщъыьэюяёєіїґӧ «°±µ»$%&‘’‚“`”„",
        punctuations="!¡'(),-.:;¿? ",
        phonemes=_phonemes + '12345',
    ),
)

# force the convertion of the custom characters to a config attribute
config.from_dict(config.to_dict())

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())


# load training samples
def formatter(root_path, meta_file=None, ignored_speakers=None):
    filename_text_dict = {}
    with open(os.path.join(root_path, meta_file), 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            filename, text = line.split('|')
            filename_text_dict[filename] = text

    items = []
    for wav_file_path in glob(f"{os.path.join(root_path, 'corpus/train')}/{folder_filter}/*.wav", recursive=True):
        wav_filename = os.path.basename(wav_file_path).split('.wav')[0]
        items.append({'audio_file': wav_file_path,
                      'text': filename_text_dict[wav_filename],
                      'speaker_name': wav_filename[5:10]})

    return items


train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
    formatter=formatter
)

# init speaker manager for multi-speaker training
# it maps speaker-id to speaker-name in the model and data-loader
CONFIG_SE_PATH = "config_se.json"
CHECKPOINT_SE_PATH = "SE_checkpoint.pth.tar"
USE_CUDA = torch.cuda.is_available()
speaker_manager = SpeakerManager(encoder_model_path=CHECKPOINT_SE_PATH, encoder_config_path=CONFIG_SE_PATH,
                                 d_vectors_file_path=d_vector_path,
                                 use_cuda=USE_CUDA)

language_manager = LanguageManager(config=config)
config.model_args.num_languages = language_manager.num_languages

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# init model
model = Vits(config, ap, tokenizer, speaker_manager, language_manager)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
if __name__ == '__main__':
    trainer.fit()
