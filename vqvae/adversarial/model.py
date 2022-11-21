import torch.nn as nn
import torch
from vqvae.modules.encdec import Encoder
from vqvae.modules.resnet import ResNet2d
from torchaudio.transforms import MelSpectrogram
from collections import namedtuple
import abc
from nnAudio.Spectrogram import CQT2010v2, STFT


class AbstractDiscriminator(nn.Module, abc.ABC):

    LossData = namedtuple("LossData", "loss pred cls acc loss_ew name")

    def __init__(self, emb_width):
        super().__init__()
        self.emb_width = emb_width
        self.fc = nn.Linear(emb_width, 2)
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

        y_weight = (y / torch.sum(y) + (1 - y) / torch.sum(1 - y))/2 if balance else 1/len(y)
        loss_ew = self.nllloss(probs, y)
        loss = torch.sum(loss_ew * y_weight)

        probs = torch.exp(probs)
        cls = torch.round(probs[:, 1])
        acc = torch.sum((cls == y) * y_weight)
        return [AbstractDiscriminator.LossData(loss, probs, cls, acc, loss_ew, self.name)]


class WavDiscriminator(AbstractDiscriminator):
    def __init__(self, input_channels, emb_width, level, downs_t, strides_t, reduce_type, **block_kwargs):
        super().__init__(emb_width)
        self.reduce_type = reduce_type
        self.encoder = Encoder(input_channels, emb_width, level + 1, downs_t, strides_t, **block_kwargs)
        self.name = f"wav_l{level + 1}"

    def encode(self, x):
        x = self.encoder(x)[-1]
        x = torch.max(x, dim=2).values if self.reduce_type == "max" else torch.mean(x, dim=2)
        return x


def get_prepr(type, n_fft, n_mels, sr, n_bins, hop_length, trainable, win_length, **params):
    if type == "mel":
        return None, MelSpectrogram(n_mels=n_mels, n_fft=n_fft, sample_rate=sr, hop_length=hop_length,
                              win_length=win_length, **params), 1, n_mels
    elif type == "fft":
        spec = STFT(n_fft=n_fft, center=True, hop_length=hop_length, win_length=win_length, sr=sr, trainable=trainable)
        return spec, lambda x: spec(x).permute(0, 3, 1, 2), 2, n_fft//2 + 1
#        return lambda x: torch.view_as_real(torch.stft(
#                input=x.squeeze(1), return_complex=True,
#                n_fft=n_fft, center=True, hop_length=hop_length, **params)).permute(0, 3, 1, 2),\
#            2, n_fft//2 + 1
    elif type == "cqt":
        spec = CQT2010v2(sr=sr, hop_length=hop_length, output_format="Complex",
                         n_bins=n_bins, norm=1, window='hann', pad_mode='constant', trainable=trainable)
        return spec, lambda x: spec(x).permute(0, 3, 1, 2), 2, n_bins
    raise Exception("Not implemented")


class FFTDiscriminator(AbstractDiscriminator):
    def __init__(self, n_fft, hop_length, window_size, sr, n_bins, reduce_type="max", pooltype="avg", leaky=1e-2,
                 res_depth=4, first_channels=32, prep_type="mel", n_mels=128, trainable_prep=False,
                 first_double_downsample=0, **params):
        prep_params = dict(n_fft=n_fft, hop_length=hop_length, win_length=window_size, trainable=bool(trainable_prep),
                           sr=sr, n_mels=n_mels, n_bins=n_bins)
        spec, feature_extract, in_channels, height = get_prepr(prep_type, **prep_params)

        encoder = ResNet2d(in_channels=in_channels, leaky=leaky, depth=res_depth, pooltype=pooltype,
                           first_channels=first_channels, first_double_downsample=first_double_downsample)
        mel_emb_width = encoder.logits_size * (height // encoder.downsample)

        super().__init__(mel_emb_width)
        self.spec = spec
        self.reduce_type = reduce_type
        self.prep_type = prep_type
        self.feature_extract = feature_extract
        self.encoder = encoder
        self.name = prep_type

    def preprocess(self, x):
        return self.feature_extract(x)

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


