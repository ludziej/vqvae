import torch
import torch.nn as nn
from typing import Union, List


class SkipConnectionsEncoder(nn.Module):
    def __init__(self, layers: [nn.Module], need_skip: Union[List[bool], bool] = True, pass_skips=False, pass_conds=True):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.pass_conds = [pass_conds] * len(layers) if isinstance(pass_conds, bool) else pass_conds
        self.need_skip = [need_skip] * len(layers) if isinstance(need_skip, bool) else need_skip
        assert len(self.need_skip) == len(self.layers)
        self.pass_skips = pass_skips

    def forward(self, x, cond=None):
        skips = []
        for layer, need_skip, pass_conds in zip(self.layers, self.need_skip, self.pass_conds):
            if pass_conds and cond is not None:
                x = layer(x, cond=cond)
            else:
                x = layer(x)
            x, skip = x if self.pass_skips and need_skip else (x, None)
            if need_skip:
                skips.append(skip if self.pass_skips else x)
        return x, skips


class SkipConnectionsDecoder(nn.Module):
    def __init__(self, layers: [nn.Module], need_skip: Union[List[bool], bool] = True, pass_conds=True):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.pass_conds = [pass_conds] * len(layers) if isinstance(pass_conds, bool) else pass_conds
        self.need_skip = [need_skip] * len(layers) if isinstance(need_skip, bool) else need_skip
        assert len(self.need_skip) == len(self.layers)

    def forward(self, x, cond=None):
        x, skips = x
        last_skip = len(skips) - 1
        for layer, need_skip, pass_conds in zip(self.layers, self.need_skip, self.pass_conds):
            if need_skip:
                assert last_skip >= 0
                x = (x, skips[last_skip])
                last_skip -= 1
            if pass_conds and cond is not None:
                x = layer(x, cond=cond)
            else:
                x = layer(x)
        return x

