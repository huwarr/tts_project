import torch
from dataclasses import dataclass

@dataclass
class MelSpectrogramConfig:
    num_mels = 80

@dataclass
class FastSpeechConfig:
    vocab_size = 300
    max_seq_len = 3000

    encoder_dim = 256
    encoder_n_layer = 4
    encoder_head = 2
    encoder_conv1d_filter_size = 1024

    decoder_dim = 256
    decoder_n_layer = 4
    decoder_head = 2
    decoder_conv1d_filter_size = 1024

    fft_conv1d_kernel = (9, 1)
    fft_conv1d_padding = (4, 0)

    predictor_filter_size = 256
    predictor_kernel_size = 3
    dropout = 0.1
    
    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3

    PAD_WORD = '<blank>'
    UNK_WORD = '<unk>'
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'


@dataclass
class TrainConfig:
    checkpoint_path = "./model_new"
    logger_path = "./logger"
    mel_ground_truth = "./mels"
    alignment_path = "./alignments"
    data_path = './data/train.txt'
    wavs_path = './data/LJSpeech-1.1/wavs'
    
    wandb_project = 'fastspeech_2'
    
    text_cleaners = ['english_cleaners']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    batch_size = 64
    epochs = 250
    n_warm_up_step = 300

    learning_rate = 1e-3
    weight_decay = 1e-6
    grad_clip_thresh = 1.0
    decay_step = [6250, 12500, 25000]

    save_step = 2000
    log_step = 5
    clear_Time = 20

    batch_expand_size = 1