import torch
from torch import nn, nn as nn
import copy


# copied from pytorch
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class ReZero(nn.Module):
    def __init__(self, fn, init=1e-3):  # originally in paper init was 0
        super().__init__()
        self.g = nn.Parameter(torch.tensor(init))
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.g


class CondProjection(nn.Module):
    def __init__(self, x_in, x_out, downsample, with_time=False):
        super().__init__()
        self.x_in = x_in
        self.x_out = x_out
        self.downsample = downsample
        self.cond_projection = nn.Conv1d(x_in, x_out, kernel_size=1)
        self.with_time = with_time
        if self.with_time:
            self.downsampling = nn.AvgPool1d(kernel_size=downsample, stride=downsample)

    def forward(self, x):
        if self.with_time:
            x = self.downsampling(x)
        x = self.cond_projection(x)
        return x


class BigGanSkip(nn.Module):
    def __init__(self, fn, ch_in, ch_out, res_scale, downsample=1, upsample=1):
        super().__init__()
        assert downsample == 1 or upsample == 1
        self.fn = fn
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.downsample = downsample
        self.upsample = upsample
        self.res_scale = res_scale
        if ch_out > ch_in:
            self.projection = nn.Conv1d(ch_in, ch_out - ch_in, 1)
        if self.upsample > 1:
            self.upsample_layer = nn.Upsample(scale_factor=upsample)
        if self.downsample > 1:
            self.downsample_layer = nn.AvgPool1d(downsample, downsample)

    def forward(self, x):
        y = self.fn(x)
        if self.ch_in > self.ch_out:
            x = x[:, :self.ch_out, :]
        if self.upsample > 1:
            x = self.upsample_layer(x)
        if self.downsample > 1:
            x = self.downsample_layer(x)
        if self.ch_in < self.ch_out:
            x = torch.cat([x, self.projection(x)], dim=1)
        return self.res_scale * x + y
