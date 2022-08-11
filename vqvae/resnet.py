import math
import torch.nn as nn
from optimization.normalization import CustomNormalization, Conv1dWeightStandardized
import torch


class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, norm_type, leaky_param, use_weight_standard, dilation=1, res_scale=1.0):
        super().__init__()
        padding = dilation
        bias = norm_type == "none"
        use_standard = norm_type != "none" and use_weight_standard
        self.resconv = nn.Sequential(
            Conv1dWeightStandardized(n_in, n_state, 3, 1, padding, dilation,
                                     bias=bias, use_standardization=use_standard),
            CustomNormalization(n_in, norm_type),
            nn.LeakyReLU(negative_slope=leaky_param),
            Conv1dWeightStandardized(n_state, n_in, 1, 1, 0, bias=bias, use_standardization=use_standard),
            CustomNormalization(n_in, norm_type),
            nn.LeakyReLU(negative_slope=leaky_param),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.resconv(x)


class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, m_conv=1.0, dilation_growth_rate=1, dilation_cycle=None, res_scale=False,
                 reverse_dilation=False, norm_type="none", leaky_param=1e-2, use_weight_standard=True,):
        super().__init__()
        get_depth = lambda depth: depth if dilation_cycle is None else depth % dilation_cycle
        blocks = [ResConv1DBlock(n_in, int(m_conv * n_in), leaky_param=leaky_param,
                                 use_weight_standard=use_weight_standard,
                                 dilation=dilation_growth_rate ** get_depth(depth), norm_type=norm_type,
                                 res_scale=1.0 if not res_scale else 1.0 / math.sqrt(n_depth))
                  for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.resblocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.resblocks(x)


class ResBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, leaky=0):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels)
        ) if downsample else nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activ = nn.ReLU() if leaky == 0 else nn.LeakyReLU(negative_slope=leaky)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.activ(self.bn1(self.conv1(x)))
        x = self.activ(self.bn2(self.conv2(x)))
        x = x + shortcut
        return self.activ(x)


class ResNet2d(nn.Module):
    def __init__(self, in_channels, first_channels=32, depth=4, leaky=1e-2):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, first_channels, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(first_channels),
            nn.ReLU()
        )
        self.encoder = nn.Sequential(*[
                nn.Sequential(
                    ResBlock2d(first_channels * 2**i, first_channels * 2**(i+1), downsample=True, leaky=leaky),
                    ResBlock2d(first_channels * 2**(i+1), first_channels * 2**(i+1), downsample=False, leaky=leaky)
                )
            for i in range(depth)])

        self.downsample = 2 ** (depth + 1)
        self.logits_size = first_channels * 2**depth

    def forward(self, x):
        x = self.layer0(x)
        return self.encoder(x)
