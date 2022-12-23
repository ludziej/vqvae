import numpy as np
from torch import nn as nn
import torch
import math
from torch import Tensor

from utils.misc import get_normal


@torch.no_grad()
def generate_fourier_features(depth, size, omega, device, offset=0):
    assert depth % 2 == 0
    position = torch.arange(size, device=device).unsqueeze(1) + offset
    omega_term = -math.log(omega) / depth
    div_term = torch.exp(torch.arange(0, depth, 2, device=device) * omega_term)
    pe = torch.stack([torch.sin(position * div_term),
                      torch.cos(position * div_term)], dim=-1).reshape(size, depth)
    return pe


@torch.no_grad()
def generate_ff_by_max_period(depth, length, period, device, offset=0):
    assert depth % 2 == 0
    h = depth // 2
    position = torch.arange(length, device=device).view(-1, 1).repeat(1, h) + offset
    level = torch.arange(h, device=device).reshape(1, -1)
    position = position * torch.pi / period * 2 ** (h - level)
    encoding = torch.stack([torch.sin(position), torch.cos(position)], dim=-1)
    encoding = encoding.reshape(length, depth)
    return encoding  # .round(decimals=3)


class FourierFeaturesPositionalEncoding(nn.Module):
    def __init__(self, depth: int, max_len: int = 10000, omega: float = 10000.0):
        super().__init__()
        self.max_len = max_len
        self.depth = depth
        self.omega = omega
        self.register_buffer('pe', generate_fourier_features(depth, size=max_len, omega=omega, device="cpu"))

    def forward(self, length: int = None, offset: int = 0, device=None, **params) -> Tensor:
        if offset + length < self.max_len:
            return self.pe[offset:offset + length]
        return generate_fourier_features(self.depth, size=length, omega=self.omega, offset=offset, device=device)


class BPMPositionalEncoding(nn.Module):
    def __init__(self, depth: int, levels_over_bit=4):
        super().__init__()
        self.depth = depth
        self.levels_over_bit = levels_over_bit

    def forward(self, bpm, length, offset, device):
        return generate_ff_by_max_period(period=bpm*2**self.levels_over_bit, device=device,
                                         depth=self.depth, length=length, offset=offset)


class TrainablePositionalEncoding(nn.Module):
    def __init__(self, input_shape, width, init_scale=1.0):
        super().__init__()
        self.input_shape = input_shape
        self.input_dims = np.prod(input_shape)
        self.pos_emb = nn.Parameter(get_normal(self.input_dims, width, std=init_scale))

    def forward(self, **params):
        return self.pos_emb
