import torch
import torch.nn as nn


class SkipConnectionsEncoder(nn.Module):
    def __init__(self, layers: [(nn.Module, bool)], pass_skips=False):
        super().__init__()
        layers, self.need_skip = zip(*layers)
        self.pass_skips = pass_skips
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        skips = []
        for layer, need_skip in zip(self.layers, self.need_skip):
            x = layer(x)
            x, skip = x if self.pass_skips else (x, None)
            if need_skip:
                skips.append(skip if self.pass_skips else x)
        return x, skips


class SkipConnectionsDecoder(nn.Module):
    def __init__(self, layers: [(nn.Module, bool)]):
        super().__init__()
        layers, self.need_skip = zip(*layers)
        self.layers = nn.ModuleList(layers)

    def forward(self, x, skips=()):
        last_skip = len(skips) - 1
        for layer, need_skip in zip(self.layers, self.need_skip):
            if need_skip:
                assert last_skip >= 0
                x = layer(x, skips[last_skip])
                last_skip -= 1
            else:
                x = layer(x)
        return x, skips

