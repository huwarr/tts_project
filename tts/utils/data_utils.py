import time
import os

import torch
from tqdm.auto import tqdm
import numpy as np
import pyworld
import torchaudio
from torchaudio.transforms import Spectrogram, MelScale
from sklearn.preprocessing import StandardScaler

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

    wav_to_spec = Spectrogram(n_fft=1024, hop_length=256, power=1)
    spec_to_mels = MelScale(n_mels=melspec_config.num_mels, n_stft=1024 // 2 + 1, sample_rate=22050, f_min=0., f_max=8000, norm='slaney', mel_scale='slaney')

    pitch_scaler = StandardScaler()
    energy_scaler = StandardScaler()

    start = time.perf_counter()
    for i in tqdm(range(len(text))):
        duration = np.load(os.path.join(
            train_config.alignment_path, str(i)+".npy"))
        
        wav_path = os.path.join(train_config.wavs_path, "LJ{}.wav".format(names[i]))
        wav, sr = torchaudio.load(wav_path)
        wav = wav.squeeze().double()
        pitch, t = pyworld.dio(wav.numpy(), sr, frame_period=256 / 22050 *1000)    # raw pitch extractor
        pitch = pyworld.stonemask(wav.numpy(), pitch, t, sr)   # pitch contour

        spec = wav_to_spec(wav)
        energy = torch.norm(spec, p='fro', dim=0)
        mel_spec = spec_to_mels(spec.float())
        mel_spec = mel_spec.transpose(-1, -2)
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

        character = text[i][0:len(text[i])-1]
        character = np.array(
            text_to_sequence(character, train_config.text_cleaners))

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)

        pitch_scaler.partial_fit(pitch.reshape(-1, 1))
        energy_scaler.partial_fit(energy.numpy().reshape(-1, 1))

        buffer.append(
            {
                "text": character, 
                "duration": duration,
                "pitch": torch.from_numpy(pitch),
                "energy": energy,
                "mel_target": mel_spec
            }
        )
    
    pitch_min = float('+inf')
    pitch_max = float('-inf')
    energy_min = float('+inf')
    energy_max = float('-inf')
    buffer_new = []
    for item in buffer:
        pitch = item['pitch']
        pitch_normalized = pitch_scaler.transform(pitch.numpy().reshape(-1, 1)).reshape(-1)
        energy = item['energy']
        energy_normalized = energy_scaler.transform(energy.numpy().reshape(-1, 1)).reshape(-1)
        item['pitch'] = torch.from_numpy(pitch_normalized)
        item['energy'] = torch.from_numpy(energy_normalized)
        buffer_new.append(item)
        pitch_min = min(pitch_min, pitch_normalized.min())
        pitch_max = max(pitch_max, pitch_normalized.max())
        energy_min = min(energy_min, energy_normalized.min())
        energy_max = max(energy_max, energy_normalized.max())

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))

    return buffer_new, pitch_min, pitch_max, energy_min, energy_max