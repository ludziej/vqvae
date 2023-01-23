import math
import torch.nn as nn
from optimization.normalization import CustomNormalization, Conv1dWeightStandardized
from vqvae.modules.skip_connections import SkipConnectionsDecoder, SkipConnectionsEncoder
import torch


class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, norm_type, leaky_param, use_weight_standard, dilation=1, concat_skip=False,
                 res_scale=1.0, num_groups=32):
        super().__init__()
        self.concat_skip = concat_skip
        padding = dilation
        bias = norm_type == "none"
        use_standard = norm_type != "none" and use_weight_standard
        blocks = [
            Conv1dWeightStandardized(n_in, n_state, 3, 1, padding, dilation,
                                     bias=bias, use_standardization=use_standard),
            CustomNormalization(n_in, norm_type, num_groups=num_groups),
            nn.LeakyReLU(negative_slope=leaky_param),
            Conv1dWeightStandardized(n_state, n_in, 1, 1, 0, bias=bias, use_standardization=use_standard),
            CustomNormalization(n_in, norm_type, num_groups=num_groups),
            nn.LeakyReLU(negative_slope=leaky_param),
        ]
        if self.concat_skip:
            blocks = [
                Conv1dWeightStandardized(n_in*2, n_in, 1, 1, 0, bias=bias, use_standardization=use_standard),
                nn.LeakyReLU(negative_slope=leaky_param),
            ] + blocks
        self.resconv = nn.Sequential(*blocks)
        self.res_scale = res_scale

    def forward(self, x):
        x, skip = x if isinstance(x, tuple) else (x, None)
        if self.concat_skip:
            assert skip is not None
            x = torch.cat([x, skip], dim=1)
            return x * self.res_scale + self.resconv(x)
        else:
            skip = skip if skip is not None else 0
            return skip + x * self.res_scale + self.resconv(x)


class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, m_conv=1.0, dilation_growth_rate=1, dilation_cycle=None, res_scale=False,
                 reverse_dilation=False, norm_type="none", leaky_param=1e-2, use_weight_standard=True,
                 get_skip=False, return_skip=False, concat_skip=False, num_groups=32):
        super().__init__()
        assert not (get_skip and return_skip)
        assert not (concat_skip and not get_skip)
        self.get_skip = get_skip
        self.return_skip = return_skip
        get_depth = lambda depth: depth if dilation_cycle is None else depth % dilation_cycle
        blocks = [ResConv1DBlock(n_in, int(m_conv * n_in), leaky_param=leaky_param,
                                 use_weight_standard=use_weight_standard, concat_skip=concat_skip,
                                 dilation=dilation_growth_rate ** get_depth(depth), norm_type=norm_type,
                                 res_scale=1.0 if not res_scale else 1.0 / math.sqrt(n_depth), num_groups=num_groups)
                  for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.resblocks = SkipConnectionsEncoder(blocks) if return_skip else \
            SkipConnectionsDecoder(blocks) if get_skip else \
                nn.Sequential(*blocks)

    def forward(self, x):
        return self.resblocks(x)


class ResBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, use_stride=True, leaky=0):
        super().__init__()
        stride = 2 if downsample and use_stride else 1  # for pooltype=max, use stride, either way use avg pool
        self.downlayer = nn.AvgPool2d(kernel_size=2, padding=0, stride=2) \
            if not use_stride and downsample else nn.Sequential()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if downsample else nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activ = nn.ReLU() if leaky == 0 else nn.LeakyReLU(negative_slope=leaky)

    def forward(self, x):
        x = self.downlayer(x)
        shortcut = self.shortcut(x)
        x = self.activ(self.bn1(self.conv1(x)))
        x = self.activ(self.bn2(self.conv2(x)))
        x = x + shortcut
        return self.activ(x)


class ResNet2d(nn.Module):
    def __init__(self, in_channels, first_channels=32, depth=4, pooltype="max", use_stride=True,
                 first_double_downsample=0, leaky=1e-2):
        super().__init__()
        self.use_stride = use_stride
        self.first_double_downsample = first_double_downsample
        self.pool = nn.MaxPool2d if pooltype == "max" else nn.AvgPool2d if pooltype == "avg" else None
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, first_channels, kernel_size=13, stride=1, padding=6, bias=False),
            self.pool(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(first_channels),
            nn.ReLU() if leaky == 0 else nn.LeakyReLU(negative_slope=leaky)
        )
        args = dict(leaky=leaky, use_stride=use_stride)
        self.encoder = nn.Sequential(*[
                nn.Sequential(
                    ResBlock2d(first_channels * 2**i, first_channels * 2**(i+1), downsample=True,
                               **args),
                    ResBlock2d(first_channels * 2**(i+1), first_channels * 2**(i+1),
                               downsample=i < self.first_double_downsample, **args)
                )
            for i in range(depth)])

        self.downsample = 2 ** (depth + 1 + self.first_double_downsample)
        self.logits_size = first_channels * 2**depth

    def forward(self, x):
        x = self.layer0(x)
        return self.encoder(x)
