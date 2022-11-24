import torch
import torch.nn as nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, mel, duration_predicted, pitch_predicted, energy_predicted, mel_target, duration_target, pitch_target, energy_target):
        mel_loss = self.mse_loss(mel, mel_target)

        duration_loss = self.mse_loss(duration_predicted, duration_target)
        pitch_loss = self.mse_loss(pitch_predicted, pitch_target)
        energy_loss = self.mse_loss(energy_predicted, energy_target)

        return mel_loss, duration_loss, pitch_loss, energy_loss