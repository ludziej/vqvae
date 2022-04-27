import math
import torch.nn as nn
import old_ml_utils.dist_adapter as dist
from old_ml_utils.checkpoint import checkpoint
import logging
from optimization.normalization import CustomNormalization


class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, norm_type, dilation=1, zero_out=False, res_scale=1.0):
        super().__init__()
        padding = dilation
        self.model = nn.Sequential(
            nn.ReLU(),
            CustomNormalization(n_in, norm_type),
            nn.Conv1d(n_in, n_state, 3, 1, padding, dilation),
            nn.ReLU(),
            CustomNormalization(n_in, norm_type),
            nn.Conv1d(n_state, n_in, 1, 1, 0),
        )
        if zero_out:
            out = self.model[-1]
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.model(x)


class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, m_conv=1.0, dilation_growth_rate=1, dilation_cycle=None, zero_out=False,
                 res_scale=False, reverse_dilation=False, checkpoint_res=False, norm_type="none"):
        super().__init__()
        def _get_depth(depth):
            if dilation_cycle is None:
                return depth
            else:
                return depth % dilation_cycle
        blocks = [ResConv1DBlock(n_in, int(m_conv * n_in),
                                 dilation=dilation_growth_rate ** _get_depth(depth),
                                 zero_out=zero_out, norm_type=norm_type,
                                 res_scale=1.0 if not res_scale else 1.0 / math.sqrt(n_depth))
                  for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.checkpoint_res = checkpoint_res
        if self.checkpoint_res == 1:
            self.blocks = nn.ModuleList(blocks)
        else:
            self.model = nn.Sequential(*blocks)

    def forward(self, x):
        if self.checkpoint_res == 1:
            for block in self.blocks:
                x = checkpoint(block, (x, ), block.parameters(), True)
            return x
        else:
            return self.model(x)
