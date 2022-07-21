import math
import torch.nn as nn
import logging
from optimization.normalization import CustomNormalization


class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, norm_type, leaky_param, dilation=1, res_scale=1.0):
        super().__init__()
        padding = dilation
        self.resconv = nn.Sequential(
            nn.Conv1d(n_in, n_state, 3, 1, padding, dilation, bias=norm_type == "none"),
            CustomNormalization(n_in, norm_type),
            nn.LeakyReLU(negative_slope=leaky_param),
            nn.Conv1d(n_state, n_in, 1, 1, 0, bias=norm_type == "none"),
            CustomNormalization(n_in, norm_type),
            nn.LeakyReLU(negative_slope=leaky_param),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.resconv(x)


class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, m_conv=1.0, dilation_growth_rate=1, dilation_cycle=None, res_scale=False,
                 reverse_dilation=False, norm_type="none", leaky_param=1e-2):
        super().__init__()
        get_depth = lambda depth: depth if dilation_cycle is None else depth % dilation_cycle
        blocks = [ResConv1DBlock(n_in, int(m_conv * n_in), leaky_param=leaky_param,
                                 dilation=dilation_growth_rate ** get_depth(depth), norm_type=norm_type,
                                 res_scale=1.0 if not res_scale else 1.0 / math.sqrt(n_depth))
                  for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.resblocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.resblocks(x)
