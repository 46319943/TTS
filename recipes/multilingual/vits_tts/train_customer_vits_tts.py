import os
from glob import glob

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

vitsArgs = VitsArgs(
    use_language_embedding=False,
    embedded_language_dim=4,
    use_speaker_embedding=True,
    d_vector_dim=512,
    use_sdp=False,
)

config = VitsConfig(
    model_args=vitsArgs,
    audio=audio_config,
    run_name="vits_vctk",
    use_speaker_embedding=True,
    batch_size=4,
    eval_batch_size=2,
    batch_group_size=0,
    num_loader_workers=2,
    num_eval_loader_workers=1,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="multilingual_cleaners",
    use_phonemes=True,
    phoneme_language="zh-cn",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    use_language_weighted_sampler=False,
    print_eval=False,
    mixed_precision=False,
    min_audio_len=32 * 256 * 4,
    max_audio_len=160000,
    output_path=output_path,
    datasets=dataset_config,
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
    for wav_file_path in glob(f"{os.path.join(root_path, 'corpus/train')}/**/*.wav", recursive=True):
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
import torch
USE_CUDA = torch.cuda.is_available()
speaker_manager = SpeakerManager(encoder_model_path=CHECKPOINT_SE_PATH, encoder_config_path=CONFIG_SE_PATH,
                                 use_cuda=USE_CUDA)
# speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
# config.model_args.num_speakers = speaker_manager.num_speakers

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


from TTS.tts.utils.synthesis import synthesis