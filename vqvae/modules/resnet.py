import math
import torch.nn as nn
from optimization.normalization import CustomNormalization, Conv1dWeightStandardized
from optimization.layers import ReZero, CondProjection
from optimization.basic_transformer import TransformerLayer
from vqvae.modules.skip_connections import SkipConnectionsDecoder, SkipConnectionsEncoder
import torch


class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, norm_type, leaky_param, use_weight_standard, dilation=1, concat_skip=False,
                 use_bias=False, res_scale=1.0, num_groups=32, rezero=False, condition_size=None, with_self_attn=False,
                 attn_heads=2, downsample=1, cond_with_time=False, alt_order=False, cond_on_attn=False,
                 rezero_in_attn=False, swish_act=False):
        super().__init__()
        self.condition_on_size = condition_size is not None
        self.concat_skip = concat_skip
        self.with_self_attn = with_self_attn
        self.rezero = rezero
        self.res_scale = res_scale
        self.cond_on_attn = cond_on_attn
        padding = dilation
        bias = norm_type == "none" or use_bias
        use_standard = norm_type != "none" and use_weight_standard
        first_in = 2 * n_in if self.concat_skip else n_in
        c_params = dict(bias=bias, use_standardization=use_standard)
        num_groups = num_groups if n_in // num_groups >= 8 else num_groups / 2  # heuristic
        conv1 = Conv1dWeightStandardized(first_in, n_state, 3, 1, padding, dilation, **c_params)
        conv2 = Conv1dWeightStandardized(n_state, n_in, 1, 1, 0, **c_params)
        norm1 = CustomNormalization(first_in if alt_order else n_in, norm_type, num_groups=num_groups)
        norm2 = CustomNormalization(n_in, norm_type, num_groups=num_groups)
        activation = nn.SiLU() if swish_act else\
            nn.LeakyReLU(negative_slope=leaky_param) if leaky_param != 0 else nn.ReLU()
        blocks = nn.Sequential(*[
            conv1, norm1, activation, conv2, norm2, activation
        ] if not alt_order else [
            norm1, conv1, activation, norm2, conv2, activation
        ])
        self.resconv = ReZero(blocks) if self.rezero else blocks

        if with_self_attn:
            attn_block = TransformerLayer(width=n_in, heads=attn_heads, seq_last=True, dropout=0, swish=swish_act)
            self.attn_block = ReZero(attn_block) if rezero_in_attn else attn_block

        if self.condition_on_size:
            self.cond_projection = CondProjection(x_in=condition_size, x_out=first_in, downsample=downsample,
                                                  with_time=cond_with_time)
            if self.cond_on_attn:
                self.cond_projection_attn = CondProjection(x_in=condition_size, x_out=n_in, downsample=downsample,
                                                           with_time=cond_with_time)

    def forward(self, x, cond=None):
        x, skip = x if isinstance(x, tuple) else (x, None)
        x_res = x if self.res_scale == 1 else x * self.res_scale
        add = 0
        if self.condition_on_size:
            assert cond is not None
            add = self.cond_projection(cond)
        if self.concat_skip:
            assert skip is not None
            x = torch.cat([x, skip], dim=1)
        else:
            add += skip if skip is not None else 0
        x = x_res + self.resconv(x + add)
        if self.with_self_attn:
            x_res = x if self.res_scale == 1 else x * self.res_scale
            add = self.cond_projection_attn(cond) if self.cond_on_attn else 0
            x = x_res + self.attn_block(x + add)
        return x


class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, m_conv=1.0, dilation_growth_rate=1, dilation_cycle=None, res_scale=1.0,
                 reverse_dilation=False, norm_type="none", leaky_param=1e-2, use_weight_standard=True, get_skip=False,
                 return_skip=False, concat_skip=False, use_bias=False, rezero=False, num_groups=32,
                 skip_connections_step=1, condition_size=None, downsample=1, with_self_attn=False,
                 cond_with_time=False, swish_act=False, rezero_in_attn=False,):
        super().__init__()
        assert not (get_skip and return_skip)
        concat_skip = concat_skip and get_skip
        self.get_skip = get_skip
        self.return_skip = return_skip
        self.dilation_cycle = dilation_cycle
        self.dilation_growth_rate = dilation_growth_rate
        skips = [d % skip_connections_step == 0 for d in range(n_depth)]
        skips = skips[::-1] if get_skip ^ reverse_dilation else skips
        blocks = [ResConv1DBlock(n_in, int(m_conv * n_in), leaky_param=leaky_param,
                                 use_weight_standard=use_weight_standard, use_bias=use_bias,
                                 concat_skip=concat_skip and get_skip and skips[depth],
                                 dilation=self.get_dilation(depth), norm_type=norm_type, rezero=rezero,
                                 res_scale=res_scale, num_groups=num_groups, condition_size=condition_size,
                                 with_self_attn=with_self_attn, downsample=downsample, cond_with_time=cond_with_time,
                                 rezero_in_attn=rezero_in_attn, swish_act=swish_act)
                  for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.resblocks = \
            SkipConnectionsEncoder(blocks, skips) if return_skip else \
            SkipConnectionsDecoder(blocks, skips) if get_skip else \
            nn.Sequential(*blocks)

    def get_dilation(self, depth):
        depth = depth if self.dilation_cycle is None else depth % self.dilation_cycle
        dilation = self.dilation_growth_rate ** depth
        return dilation

    def forward(self, x, cond=None):
        return self.resblocks(x) if cond is None else self.resblocks(x, cond=cond)


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
