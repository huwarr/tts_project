import os

import torch
import numpy as np
from tqdm.auto import tqdm
import torchaudio

import waveglow
import text
import audio
import utils

from tts.configs.all_configs import MelSpectrogramConfig, FastSpeechConfig, TrainConfig
from tts.model.fastspeech2 import FastSpeech2


def synthesis(train_config, model, text, duration_alpha=1.0, pitch_alpha=1.0, energy_alpha=1.0):
    text = np.array(text)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(train_config.device)
    src_pos = torch.from_numpy(src_pos).long().to(train_config.device)
    
    with torch.no_grad():
        mel = model.forward(sequence, src_pos, duration_alpha=duration_alpha, pitch_alpha=duration_alpha, energy_alpha=energy_alpha)
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)


def get_data(train_config):
    tests = [ 
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space"
    ]
    data_list = list(text.text_to_sequence(test, train_config.text_cleaners) for test in tests)

    return data_list


def run_full_synthesis(checkpoint_path='checkpoint.pth.tar', logger=None):
    train_config = TrainConfig()
    WaveGlow = utils.get_WaveGlow()
    WaveGlow = WaveGlow.to(train_config.device)

    model_config = FastSpeechConfig()
    mel_config = MelSpectrogramConfig()
    model = FastSpeech2(model_config, mel_config)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cuda:0')['model'])
    model = model.eval()
    model = model.to(train_config.device)

    data_list = get_data(train_config)

    # speed
    for speed in [0.8, 1., 1.2]:
        for i, phn in tqdm(enumerate(data_list)):
            mel, mel_cuda = synthesis(train_config, model, phn, duration_alpha=speed)
            
            os.makedirs("results", exist_ok=True)
            
            path = f"results/speed={speed}_{i}.wav" if speed != 1. else f"results/usual_audio_{i}.wav"
            audio.tools.inv_mel_spec(
                mel, path
            )
            
            waveglow.inference.inference(
                mel_cuda, WaveGlow,
                path
            )

            if logger is not None:
                wav, sr = torchaudio.load(path).float()
                logger.add_audio(f"speed={speed}_{i}", wav, sample_rate=sr)

    # pitch
    for pitch in [0.8, 1.2]:
        for i, phn in tqdm(enumerate(data_list)):
            mel, mel_cuda = synthesis(train_config, model, phn, pitch_alpha=pitch)
            
            os.makedirs("results", exist_ok=True)
            
            audio.tools.inv_mel_spec(
                mel, f"results/pitch={pitch}_{i}.wav"
            )
            
            waveglow.inference.inference(
                mel_cuda, WaveGlow,
                f"results/pitch={pitch}_{i}.wav"
            )

            if logger is not None:
                wav, sr = torchaudio.load(f"results/pitch={pitch}_{i}.wav").float()
                logger.add_audio(f"pitch={pitch}_{i}", wav, sample_rate=sr)

    # energy
    for energy in [0.8, 1.2]:
        for i, phn in tqdm(enumerate(data_list)):
            mel, mel_cuda = synthesis(train_config, model, phn, energy_alpha=energy)
            
            os.makedirs("results", exist_ok=True)
            
            audio.tools.inv_mel_spec(
                mel, f"results/energy={energy}_{i}.wav"
            )
            
            waveglow.inference.inference(
                mel_cuda, WaveGlow,
                f"results/energy={energy}_{i}.wav"
            )

            if logger is not None:
                wav, sr = torchaudio.load(f"results/energy={energy}_{i}.wav").float()
                logger.add_audio(f"energy={energy}_{i}", wav, sample_rate=sr)

    # all together
    for alpha in [0.8, 1.2]:
        for i, phn in tqdm(enumerate(data_list)):
            mel, mel_cuda = synthesis(train_config, model, phn, duration_alpha=alpha, pitch_alpha=alpha, energy_alpha=alpha)
            
            os.makedirs("results", exist_ok=True)
            
            audio.tools.inv_mel_spec(
                mel, f"results/speed={alpha}_pitch={alpha}_energy={alpha}_{i}.wav"
            )
            
            waveglow.inference.inference(
                mel_cuda, WaveGlow,
                f"results/speed={alpha}_pitch={alpha}_energy={alpha}_{i}.wav"
            )

            if logger is not None:
                wav, sr = torchaudio.load(f"results/speed={alpha}_pitch={alpha}_energy={alpha}_{i}.wav").float()
                logger.add_audio(f"speed={alpha}_pitch={alpha}_energy={alpha}_{i}", wav, sample_rate=sr)