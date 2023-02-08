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
