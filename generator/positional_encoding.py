import numpy as np
from torch import nn as nn
import torch
import math

from generator.conditioner import get_normal


class PositionEmbedding(nn.Module):
    def __init__(self, input_shape, width, init_scale=1.0):
        super().__init__()
        self.input_shape = input_shape
        self.input_dims = np.prod(input_shape)
        self.pos_emb = nn.Parameter(get_normal(self.input_dims, width, std=0.01 * init_scale))

    def forward(self):
        return self.pos_emb


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000, omega: float = 10000.0):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        omega_term = -math.log(omega) / d_model
        div_term = torch.exp(torch.arange(0, d_model, 2) * omega_term)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, length: int = None) -> Tensor:
        return self.pe[:length]
