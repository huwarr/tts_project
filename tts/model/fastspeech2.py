import torch
from torch import nn
import torch.nn.functional as F

from tts.model.fastspeech_utilities import Encoder, Decoder

def create_alignment(base_mat, duration_predictor_output):
    N, L = duration_predictor_output.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_predictor_output[i][j]):
                base_mat[i][count+k][j] = 1
            count = count + duration_predictor_output[i][j]
    return base_mat

def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, 1, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask


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
            x = x.unsqueeze(0)
        return x


class VarianceAdaptor(nn.Module):
    def __init__(self, model_config, pitch_min, pitch_max, energy_min, energy_max):
        super().__init__()

        self.duration_predictor = PredictorBlock(model_config)
        self.pitch_predictor = PredictorBlock(model_config)
        self.energy_predictor = PredictorBlock(model_config)

        self.pitch_embed = nn.Embedding(256, model_config.encoder_dim)
        self.energy_embed = nn.Embedding(256, model_config.encoder_dim)

        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.energy_min = energy_min
        self.energy_max = energy_max
    
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

    def forward(self, x, duration_alpha=1.0, pitch_alpha=1.0, energy_alpha=1.0, length_target=None, mel_max_length=None):
        log_duration = self.duration_predictor(x)

        if length_target is not None:
            x = self.LR(x, length_target, mel_max_length)
            pos = torch.exp(log_duration)
            if mel_max_length:
                pos = F.pad(pos, (0, mel_max_length-pos.size(1)))
        else:
            duration = (torch.exp(log_duration) * duration_alpha + 0.5).int()
            x = self.LR(x, duration, mel_max_length)
            pos = torch.stack(
                [torch.Tensor([i+1 for i in range(x.size(1))])]
            ).long().to(x.device)

        pitch = self.pitch_predictor(x) * pitch_alpha
        buckets = torch.linspace(torch.log(torch.tensor(self.pitch_min) + 1).item(), torch.log(torch.tensor(self.pitch_max) + 1).item(), 256).to(x.device)
        pitch_quantized = torch.bucketize(torch.log(pitch + 1), buckets[:-1]).to(x.device)
        
        energy = self.energy_predictor(x) * energy_alpha
        buckets = torch.linspace(torch.tensor(self.energy_min), torch.tensor(self.energy_max), 256).to(x.device)
        energy_quantized = torch.bucketize(energy, buckets[:-1]).to(x.device)


        out = self.pitch_embed(pitch_quantized) + self.energy_embed(energy_quantized) + x
        return out, pos, log_duration, pitch, energy


class FastSpeech2(nn.Module):
    def __init__(self, model_config, mel_config, pitch_min, pitch_max, energy_min, energy_max):
        super().__init__()

        self.encoder = Encoder(model_config)
        self.decoder = Decoder(model_config)

        self.variance_adaptor = VarianceAdaptor(model_config, pitch_min, pitch_max, energy_min, energy_max)
    
        self.mel_linear = nn.Linear(model_config.decoder_dim, mel_config.num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, src_pos, mel_pos=None, length_target=None, mel_max_length=None, duration_alpha=1.0, pitch_alpha=1.0, energy_alpha=1.0):
        x, non_pad_mask = self.encoder(src_seq, src_pos)
        
        if self.training:
            x, pos, log_duration, pitch, energy = self.variance_adaptor(x, duration_alpha, pitch_alpha, energy_alpha, length_target, mel_max_length)
            x = self.decoder(x, mel_pos)
            x = self.mask_tensor(x, mel_pos, mel_max_length)
            x = self.mel_linear(x)
            return x, log_duration, pitch, energy
        else:
            x, mel_pos, log_duration, pitch, energy = self.variance_adaptor(x, duration_alpha, pitch_alpha, energy_alpha)
            x = self.decoder(x, mel_pos)
            x = self.mel_linear(x)
            return x
