import torch
from torch import nn
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
    def __init__(self, fn, init=0.):  # originally init was 1e-3
        super().__init__()
        self.g = nn.Parameter(torch.tensor(init))
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.g
