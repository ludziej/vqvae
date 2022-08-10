import torch.nn as nn
import torch
from vqvae.encdec import Encoder
from vqvae.resnet import ResNet2d
from torchaudio.transforms import MelSpectrogram
import abc


class AbstractDiscriminator(nn.Module, abc.ABC):
    def __init__(self, emb_width):
        super().__init__()
        self.emb_width = emb_width
        self.fc = nn.Linear(emb_width, 2)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.nllloss = nn.NLLLoss(reduction="none")

    @abc.abstractmethod
    def encode(self, x):
        """"""

    def forward(self, x):
        self.encode(x)
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
        return [(loss, probs, cls, acc, loss_ew)]


class WavDiscriminator(AbstractDiscriminator):
    def __init__(self, input_channels, emb_width, level, downs_t, strides_t, wav_reduce_type="max", **block_kwargs):
        super().__init__(emb_width)
        self.reduce_type = wav_reduce_type
        self.encoder = Encoder(input_channels, emb_width, level + 1, downs_t, strides_t, **block_kwargs)

    def encode(self, x):
        x = self.encoder(x)[-1]
        x = torch.max(x, dim=2).values if self.reduce_type == "max" else torch.mean(x, dim=2)
        return x


class MelDiscriminator(AbstractDiscriminator):
    def __init__(self, n_fft, hop_length, window_size, sr, mel_emb_width, reduce_type="max", leaky=1e-2, res_depth=4,
                 first_channels=32, **params):
        super().__init__(mel_emb_width)
        self.reduce_type = reduce_type
        self.melspec = MelSpectrogram(n_fft=n_fft, hop_length=hop_length, win_length=window_size, sample_rate=sr)
        self.encoder = ResNet2d(in_channels=2, leaky=leaky, depth=res_depth, first_channels=first_channels)

    def preprocess(self, x):
        return self.melspec.forward(x)

    def encode(self, x):
        x = self.preprocess(x)
        x = self.encoder(x)
        x = torch.max(x, dim=-1)
        return x


class JoinedDiscriminator(nn.Module):
    def __init__(self, n_fft, hop_length, window_size, sr, emb_width, mel_emb_width, reduce_type="max", **block_kwargs):
        super().__init__()
        self.meldiscriminator = MelDiscriminator(n_fft=n_fft, hop_length=hop_length, window_size=window_size, sr=sr,
                                                 mel_emb_width=mel_emb_width, )
        self.wavdiscriminator = WavDiscriminator(reduce_type=reduce_type, emb_width=emb_width, **block_kwargs)

    def forward(self, x):
        return (self.meldiscriminator(x) + self.wavdiscriminator(x))/2

    def calculate_loss(self, x, y, balance=True):
        stats1 = self.meldiscriminator.calculate_loss(x, y, balance)
        stats2 = self.wavdiscriminator.calculate_loss(x, y, balance)
        return (*stats1, "wav_discriminator"), (*stats2, "mel_discriminator")


