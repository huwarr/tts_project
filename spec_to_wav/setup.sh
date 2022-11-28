#!/bin/env bash
pip install -r requirements.txt
pip install gdown==4.5.4 --no-cache-dir

#download Waveglow
gdown https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx
mkdir -p waveglow/pretrained_model/
mv waveglow_256channels_ljs_v2.pt waveglow/pretrained_model/waveglow_256channels.pt

# we will use waveglow code, data and audio preprocessing from this repo
git clone https://github.com/xcmyz/FastSpeech.git
mv FastSpeech/text .
mv FastSpeech/audio .
mv FastSpeech/waveglow/* waveglow/
mv FastSpeech/utils.py .
mv FastSpeech/glow.py .

# Download FastSpeech2 checkpoint
gdown https://drive.google.com/uc?id=1zTGB06g-cj9DiNsAqoje9cq5OVLBqU_u