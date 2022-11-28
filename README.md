# TTS project

An attempt to implement and train `FastSpeech2`, vastly reusing implementation of `FastSpeech` from the seminar.

[checkpoint](https://drive.google.com/file/d/1zTGB06g-cj9DiNsAqoje9cq5OVLBqU_u/view?usp=share_link)

**synthesised audio:** in `generated samples` folder or [here](https://drive.google.com/drive/folders/1Iu8Yt7QHnnrsVLApXKrqYzSQtnHAwvMD?usp=share_link)

## Installation guide

First, clone this repository to get access to the code:

`git clone https://github.com/huwarr/tts_project.git`

`cd tts_project`

## Synthesising audio

Run `setup.sh` script to download requirements and all the necessary files for synthesisng audio, including checkpoint:

`sh setup.sh`

Run python file to synthesis audio samples using checkpoint:

`python get_wav.py`

You might see several warnings, related to WaveGlow, but they will not affect success of this script's execution.

When everything is done (it takes approximately 3 minutes), you can view samples in `results` folder :)

## Training

Firts, load training data with running the script:

`sh load_data.sh`

Install dependencies:

`pip install -r requirements.txt`

Run training script:

`python train.py`

You will see intermideate checkpoints of the model in `model_new` folder, the final checkpoint as `checkpoint.pth.tar` in the root of this repository, and samples, synthesised with the final checkpoint, in `results` folder.

## Sources

1. [[V1] FastSpeech 2: Fast and High-Quality End-to-End Text-to-Speech](https://arxiv.org/pdf/2006.04558v1)
2. [[V8] FastSpeech 2: Fast and High-Quality End-to-End Text-to-Speech](https://arxiv.org/pdf/2006.04558)
