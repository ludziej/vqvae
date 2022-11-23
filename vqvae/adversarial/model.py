import torch.nn as nn
import torch
from vqvae.modules.encdec import Encoder
from vqvae.modules.resnet import ResNet2d
from torchaudio.transforms import MelSpectrogram
from collections import namedtuple
import abc
from nnAudio.Spectrogram import CQT2010v2, STFT
import torchvision
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from utils.misc import get_normal


class AbstractDiscriminator(nn.Module, abc.ABC):

    LossData = namedtuple("LossData", "loss pred cls acc loss_ew name")

    def __init__(self, emb_width, classify_each_level, levels, can_plot=False):
        super().__init__()
        self.emb_width = emb_width
        self.can_plot = can_plot
        self.classify_each_level = classify_each_level
        self.final_lvls = levels + 1 if classify_each_level else 2
        self.fc = nn.Linear(emb_width, self.final_lvls)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.nllloss = nn.NLLLoss(reduction="none")
        self.name = "discriminator"

    @abc.abstractmethod
    def encode(self, x):
        """"""

    def forward(self, x):
        x = self.encode(x)
        logits = self.fc(x)
        probs = self.logsoftmax(logits)
        return probs

    def calculate_loss(self, x, y, balance=True):
        probs = self.forward(x)

        is_fake = (y != 0).long()
        if balance and not self.classify_each_level and torch.sum(is_fake) > 0 and torch.sum(1 - is_fake) > 0:
            y_weight = (is_fake / torch.sum(is_fake) + (1 - is_fake) / torch.sum(1 - is_fake))/2
        else:
            y_weight = 1 / len(y)
        loss_ew = self.nllloss(probs, y)
        loss = torch.sum(loss_ew * y_weight)

        probs = torch.exp(probs)
        cls = probs.argmax(dim=-1)
        acc = torch.sum((cls == y) * y_weight)
        return [AbstractDiscriminator.LossData(loss, probs, cls, acc, loss_ew, self.name)]


class WavDiscriminator(AbstractDiscriminator):
    def __init__(self, input_channels, emb_width, level, downs_t, strides_t, reduce_type, classify_each_level, levels, **block_kwargs):
        super().__init__(emb_width, classify_each_level, levels)
        self.reduce_type = reduce_type
        self.encoder = Encoder(input_channels, emb_width, level + 1, downs_t, strides_t, **block_kwargs)
        self.name = f"wav_l{level + 1}"

    def encode(self, x):
        x = self.encoder(x)[-1]
        x = torch.max(x, dim=2).values if self.reduce_type == "max" else torch.mean(x, dim=2)
        return x


def post_process_fft(spec, use_amp, use_log_scale, bins):
    assert not (use_log_scale and not use_amp)  # cannot use log scale of a signal without phase discarded
    extract = lambda x: spec(x).permute(0, 3, 1, 2)
    return spec, extract, 1 if use_amp else 2, bins


def get_prepr(type, n_fft, n_mels, sr, n_bins, hop_length, trainable, win_length, use_amp, use_log_scale, **params):
    if type == "mel":
        return None, MelSpectrogram(n_mels=n_mels, n_fft=n_fft, sample_rate=sr, hop_length=hop_length,
                              win_length=win_length, **params), 1, n_mels
    elif type == "fft":
        spec = STFT(n_fft=n_fft, freq_bins=n_fft//2, center=True, hop_length=hop_length, sr=sr, trainable=trainable)
        return post_process_fft(spec, use_amp, use_log_scale, n_fft // 2)
    elif type == "cqt":
        spec = CQT2010v2(sr=sr, hop_length=hop_length, output_format="Complex",
                         n_bins=n_bins, norm=1, window='hann', pad_mode='constant', trainable=trainable)
        return post_process_fft(spec, use_amp, use_log_scale, n_bins)
    raise Exception("Not implemented")


class FFTDiscriminator(AbstractDiscriminator):
    def __init__(self, n_fft, hop_length, window_size, sr, n_bins, classify_each_level, levels, reduce_type="max",
                 pooltype="avg", leaky=1e-2, res_depth=4, first_channels=32, prep_type="mel", n_mels=128,
                 trainable_prep=False, pos_enc_size=0, first_double_downsample=0, use_stride=True, use_amp=True,
                 use_log_scale=True, **params):
        prep_params = dict(n_fft=n_fft, hop_length=hop_length, win_length=window_size, trainable=bool(trainable_prep),
                           sr=sr, n_mels=n_mels, n_bins=n_bins, use_amp=use_amp, use_log_scale=use_log_scale)
        spec, feature_extract, in_channels, height = get_prepr(prep_type, **prep_params)

        encoder = ResNet2d(in_channels=in_channels + pos_enc_size, leaky=leaky, depth=res_depth, pooltype=pooltype,
                           use_stride=use_stride, first_channels=first_channels,
                           first_double_downsample=first_double_downsample)
        mel_emb_width = encoder.logits_size * (max(height // encoder.downsample, 1))

        super().__init__(mel_emb_width, classify_each_level, levels, can_plot=True)
        self.pos_enc_size = pos_enc_size
        self.use_log_scale = use_log_scale
        self.use_amp = use_amp
        self.prep_params = prep_params
        self.spec = spec
        self.reduce_type = reduce_type
        self.prep_type = prep_type
        self.feature_extract = feature_extract
        self.encoder = encoder
        self.name = prep_type
        self.height = height
        if self.pos_enc_size > 0:
            self.pos_enc = nn.Parameter(get_normal(height, self.pos_enc_size, std=0.01))

    def get_plot(self, image):
        fft = self.preprocess(image, scale=False)
        ampl = (fft[:, 0]**2 + fft[:, 1]**2)**(1/2) if not self.use_amp else fft.squeeze(1)

        fig, ax = plt.subplots()
        data = ampl.detach()[0].cpu().numpy()
        dbb = librosa.amplitude_to_db(data, ref=np.max)
        y_type = 'cqt_hz' if self.prep_type == "cqt" else "linear"
        img = librosa.display.specshow(dbb, x_axis='time', y_axis=y_type, ax=ax,
                                       hop_length=self.prep_params["hop_length"],
                                       sr=self.prep_params["sr"])
        fig.colorbar(img, ax=ax)
        fig.tight_layout()
        fig.canvas.draw()
        X = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        return X

    def preprocess(self, x, scale=True):
        x = self.feature_extract(x)
        if self.use_amp:
            x = torch.sum(x**2, dim=1, keepdim=True)**(1/2)
        if self.use_log_scale and scale:
            x = torch.log(x + 1e8)
        if self.pos_enc_size > 0:
            pos_enc = self.pos_enc.view(1, self.pos_enc_size, self.height, 1).repeat(x.shape[0], 1, 1, x.shape[3])
            x = torch.cat([x, pos_enc], dim=1)
        return x

    def encode(self, x):
        x = self.preprocess(x)
        x = self.encoder(x)
        x = torch.max(x, dim=-1).values if self.reduce_type == "max" else torch.mean(x, dim=-1)
        x = x.reshape(x.shape[0], self.emb_width)
        return x


class JoinedDiscriminator(nn.Module):
    def __init__(self, n_fft, hop_length, window_size, sr, emb_width, reduce_type, **params):
        super().__init__()
        self.meldiscriminator = FFTDiscriminator(n_fft=n_fft, hop_length=hop_length, window_size=window_size, sr=sr,
                                                 **params)
        self.wavdiscriminator = WavDiscriminator(reduce_type=reduce_type, emb_width=emb_width, **params)

    def forward(self, x):
        return (self.meldiscriminator(x) + self.wavdiscriminator(x))/2

    def calculate_loss(self, x, y, balance=True):
        stats1 = self.meldiscriminator.calculate_loss(x, y, balance)
        stats2 = self.wavdiscriminator.calculate_loss(x, y, balance)
        return stats1 + stats2



#        return lambda x: torch.view_as_real(torch.stft(
#                input=x.squeeze(1), return_complex=True,
#                n_fft=n_fft, center=True, hop_length=hop_length, **params)).permute(0, 3, 1, 2),\
#            2, n_fft//2 + 1