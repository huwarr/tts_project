import torch
from torch import nn

from tts.model.fastspeech_utilities import Encoder, Decoder


class PredictorBlock(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        input_size = model_config.encoder_dim
        filter_size = model_config.predictor_filter_size
        kernel = model_config.predictor_kernel_size
        conv_output_size = model_config.predictor_filter_size
        dropout = model_config.dropout

        self.conv_1 = nn.Sequential(
            nn.Conv1d(input_size, filter_size, kernel_size=kernel, padding=1),
            nn.ReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(input_size, filter_size, kernel_size=kernel, padding=1),
            nn.ReLU()
        )

        self.ln_1 = nn.Sequential(
            nn.LayerNorm(filter_size),
            nn.Dropout(dropout)
        )
        self.ln_2 = nn.Sequential(
            nn.LayerNorm(filter_size),
            nn.Dropout(dropout)
        )

        self.fc = nn.Linear(conv_output_size, 1)
    
    def forward(self, x):
        x = self.conv_1(x.transpose(-1, -2))
        x = self.ln_1(x.transpose(-1, -2))
        x = self.conv_2(x.transpose(-1, -2))
        x = self.ln_2(x.transpose(-1, -2))
        x = self.fc(x)
        x = x.squeeze()
        if not self.training:
            # ????
            x = x.unsqueeze(0)
        return x


class VarianceAdaptor(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.duration_predictor = PredictorBlock(model_config)
        self.pitch_predictor = PredictorBlock(model_config)
        self.energy_predictor = PredictorBlock(model_config)
    
    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration = self.duration_predictor(x)

        if target is not None:
            x = self.LR(x, target, mel_max_length)
            pos = duration
        else:
            duration = (duration * alpha + 0.5).int()
            x = self.LR(x, duration)
            pos = torch.stack(
                [torch.Tensor([i+1 for i in range(output.size(1))])]
            ).long().to(x.device)

        pitch = self.pitch_predictor(x)
        energy = self.energy_predictor(x)
        out = pitch + energy + x
        return out, pos, duration, pitch, energy


class FastSpeech2(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.encoder = Encoder(model_config)
        self.decoder = Decoder(model_config)

        self.variance_adaptor = VarianceAdaptor(model_config)
    
    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, length_target=None, alpha=1.0):
        x, non_pad_mask = self.encoder(src_seq, src_pos)

        x, mel_pos, duration, pitch, energy = self.variance_adaptor(x, alpha, target,  length_target, mel_max_length)
        x = self.decoder(x, mel_pos)
        # don't forget to take log(duration) when estimating MSE
        return x, duration, pitch, energy