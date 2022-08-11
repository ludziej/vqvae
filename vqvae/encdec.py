import torch as t
import torch.nn as nn
from vqvae.resnet import Resnet1D
from old_ml_utils.misc import assert_shape


class EncoderConvBlock(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t,
                 stride_t, width, depth, m_conv, norm_type,
                 dilation_growth_rate=1, dilation_cycle=None,
                 res_scale=False, leaky_param=1e-2, use_weight_standard=True, **params):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        if down_t > 0:
            for i in range(down_t):
                block = nn.Sequential(
                    nn.Conv1d(input_emb_width if i == 0 else width, width, filter_t, stride_t, pad_t),
                    Resnet1D(width, depth, m_conv, dilation_growth_rate, dilation_cycle, res_scale,
                             norm_type=norm_type, leaky_param=leaky_param, use_weight_standard=use_weight_standard),
                )
                blocks.append(block)
            block = nn.Conv1d(width, output_emb_width, 3, 1, 1)
            blocks.append(block)
        self.encode_block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.encode_block(x)


class DecoderConvBock(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t,
                 stride_t, width, depth, m_conv, norm_type, dilation_growth_rate=1, dilation_cycle=None,
                 res_scale=False, reverse_decoder_dilation=False, leaky_param=1e-2, use_weight_standard=True, **params):
        super().__init__()
        blocks = []
        if down_t > 0:
            filter_t, pad_t = stride_t * 2, stride_t // 2
            block = nn.Conv1d(output_emb_width, width, 3, 1, 1)
            blocks.append(block)
            for i in range(down_t):
                block = nn.Sequential(
                    Resnet1D(width, depth, m_conv, dilation_growth_rate, dilation_cycle, leaky_param=leaky_param,
                             norm_type=norm_type, res_scale=res_scale, reverse_dilation=reverse_decoder_dilation,
                             use_weight_standard=use_weight_standard),
                    nn.ConvTranspose1d(width, input_emb_width if i == (down_t - 1) else width, filter_t, stride_t, pad_t),
                )
                blocks.append(block)
        self.decoder_block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.decoder_block(x)


class Encoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, levels, downs_t,
                 strides_t, **block_kwargs):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels
        self.downs_t = downs_t
        self.strides_t = strides_t

        block_kwargs_copy = dict(**block_kwargs)
        if 'reverse_decoder_dilation' in block_kwargs_copy:
            del block_kwargs_copy['reverse_decoder_dilation']

        self.level_blocks = nn.ModuleList([
            EncoderConvBlock(input_emb_width if level == 0 else output_emb_width,
                             output_emb_width, down_t, stride_t, **block_kwargs_copy)
            for level, down_t, stride_t in zip(range(self.levels), downs_t, strides_t)])

    def forward(self, x):
        N, T = x.shape[0], x.shape[-1]
        emb = self.input_emb_width
        assert_shape(x, (N, emb, T))
        xs = []

        # 64, 32, ...
        iterator = zip(range(self.levels), self.downs_t, self.strides_t)
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            emb, T = self.output_emb_width, T // (stride_t ** down_t)
            assert_shape(x, (N, emb, T))
            xs.append(x)

        return xs


class Decoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, levels, downs_t,
                 strides_t, **block_kwargs):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels
        self.downs_t = downs_t
        self.strides_t = strides_t

        self.level_blocks = nn.ModuleList(
            DecoderConvBock(output_emb_width, output_emb_width, down_t, stride_t, **block_kwargs)
            for level, down_t, stride_t in zip(range(self.levels), downs_t, strides_t))

        self.out = nn.Conv1d(output_emb_width, input_emb_width, 3, 1, 1)

    def forward(self, xs, all_levels=True):
        if all_levels:
            assert len(xs) == self.levels
        else:
            assert len(xs) == 1
        x = xs[-1]
        N, T = x.shape[0], x.shape[-1]
        emb = self.output_emb_width
        assert_shape(x, (N, emb, T))

        # 32, 64 ...
        iterator = reversed(list(zip(range(self.levels), self.downs_t, self.strides_t)))
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            emb, T = self.output_emb_width, T * (stride_t ** down_t)
            assert_shape(x, (N, emb, T))
            if level != 0 and all_levels:
                x = x + xs[level - 1]

        x = self.out(x)
        return x
