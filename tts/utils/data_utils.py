import time
import os

import torch
from tqdm.auto import tqdm
import numpy as np
import pyworld
import torchaudio
from torchaudio.transforms import Spectrogram, MelScale

from text import text_to_sequence


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt

def get_data_to_buffer(train_config, melspec_config):
    buffer = list()
    text = process_text(train_config.data_path)
    names = sorted(list(map(lambda x: x[2:-4], os.listdir(train_config.wavs_path))))

    wav_to_spec = Spectrogram(n_fft=1024, hop_length=256)
    spec_to_mels = MelScale(n_mels=melspec_config.num_mels, sample_rate=22050, n_stft=1024 // 2 + 1)

    start = time.perf_counter()
    for i in tqdm(range(len(text))):
        duration = np.load(os.path.join(
            train_config.alignment_path, str(i)+".npy"))
        
        wav_path = os.path.join(train_config.wavs_path, "LJ{}.wav".format(names[i]))
        wav, sr = torchaudio.load(wav_path)
        wav = wav.squeeze().double()
        pitch, t = pyworld.dio(wav.numpy(), sr, frame_period=11.6)    # raw pitch extractor
        pitch = pyworld.stonemask(wav.numpy(), pitch, t, sr)   # pitch contour

        spec = wav_to_spec(wav)
        energy = torch.norm(spec, p='fro', dim=1)
        mel_spec = spec_to_mels(spec.float())
        mel_spec = mel_spec.transpose(-1, -2)

        character = text[i][0:len(text[i])-1]
        character = np.array(
            text_to_sequence(character, train_config.text_cleaners))

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)

        buffer.append(
            {
                "text": character, 
                "duration": duration,
                "pitch": torch.from_numpy(pitch),
                "energy": energy,
                "mel_target": mel_spec
            }
        )

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))

    return buffer