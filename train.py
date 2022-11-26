import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler  import OneCycleLR
from tqdm.auto import tqdm

from tts.configs.all_configs import MelSpectrogramConfig, FastSpeechConfig, TrainConfig
from tts.utils.data_utils import get_data_to_buffer
from tts.dataset.buffer_dataset import BufferDataset
from tts.collate_fn.collate import collate_fn
from tts.model.fastspeech2 import FastSpeech2
from tts.loss.fastspeech_loss import FastSpeechLoss
from tts.logger.wandb_logger import WanDBWriter
from spec_to_wav.get_wav import run_full_synthesis


# fix seed for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# define configs
mel_config = MelSpectrogramConfig()
model_config = FastSpeechConfig()
train_config = TrainConfig()


# define dataloader
buffer = get_data_to_buffer(train_config, mel_config)
dataset = BufferDataset(buffer)

training_loader = DataLoader(
    dataset,
    batch_size=train_config.batch_expand_size * train_config.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=0
)

# training essentials
model = FastSpeech2(model_config, mel_config)
model = model.to(train_config.device)

fastspeech_loss = FastSpeechLoss()
current_step = 0

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_config.learning_rate,
    betas=(0.9, 0.98),
    eps=1e-9)

scheduler = OneCycleLR(optimizer, **{
    "steps_per_epoch": len(training_loader) * train_config.batch_expand_size,
    "epochs": train_config.epochs,
    "anneal_strategy": "cos",
    "max_lr": train_config.learning_rate,
    "pct_start": 0.1
})

# logger
logger = WanDBWriter(train_config)


# training loop
tqdm_bar = tqdm(total=train_config.epochs * len(training_loader) * train_config.batch_expand_size - current_step)

for epoch in range(train_config.epochs):
    for i, batchs in enumerate(training_loader):
        for j, db in enumerate(batchs):
            current_step += 1
            tqdm_bar.update(1)
            
            logger.set_step(current_step)

            # Get Data
            character = db["text"].long().to(train_config.device)
            mel_target = db["mel_target"].float().to(train_config.device)
            duration = db["duration"].int().to(train_config.device)
            pitch = db["pitch"].int().to(train_config.device)
            energy = db["energy"].int().to(train_config.device)
            mel_pos = db["mel_pos"].long().to(train_config.device)
            src_pos = db["src_pos"].long().to(train_config.device)
            max_mel_len = db["mel_max_len"]

            # Forward
            mel_output, log_duration_output, pitch_output, energy_output = model(
                character,
                src_pos,
                mel_pos=mel_pos,
                mel_max_length=max_mel_len,
                length_target=duration
            )

            # Cal Loss
            mel_loss, duration_loss, pitch_loss, energy_loss = fastspeech_loss(
                mel_output,
                log_duration_output,
                pitch_output,
                energy_output,
                mel_target,
                torch.log(duration + 1),
                pitch,
                energy
            )
            total_loss = mel_loss + duration_loss + pitch_loss + energy_loss

            # Logger
            t_l = total_loss.detach().cpu().numpy()
            m_l = mel_loss.detach().cpu().numpy()
            d_l = duration_loss.detach().cpu().numpy()
            p_l = pitch_loss.detach().cpu().numpy()
            e_l = energy_loss.detach().cpu().numpy()

            logger.add_scalar("duration_loss", d_l)
            logger.add_scalar("pitch_loss", p_l)
            logger.add_scalar("energy_loss", e_l)
            logger.add_scalar("mel_loss", m_l)
            logger.add_scalar("total_loss", t_l)

            # Backward
            total_loss.backward()

            # Clipping gradients to avoid gradient explosion
            nn.utils.clip_grad_norm_(
                model.parameters(), train_config.grad_clip_thresh
            )
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if current_step % train_config.save_step == 0:
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                )}, os.path.join(train_config.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                print("save model at step %d ..." % current_step)

                run_full_synthesis(checkpoint_path='checkpoint_%d.pth.tar' % current_step, logger=logger)


# save checkpoint of the trained model
torch.save(
    {'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
    'checkpoint.pth.tar'
)


logger.finish_wandb_run()