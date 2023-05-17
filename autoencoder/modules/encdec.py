import itertools
import math

import torch.nn as nn

from optimization.layers import BigGanSkip
from autoencoder.modules.resnet import Resnet1D
from utils.old_ml_utils.misc import assert_shape
from utils.misc import clamp
from autoencoder.modules.skip_connections import SkipConnectionsDecoder, SkipConnectionsEncoder


class EncoderConvBlock(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t, stride_t, skip_connections, width, depth,
                 norm_type, res_scale=False, num_groups=32, rezero=False, channel_increase=1, condition_size=None,
                 self_attn_from=None, biggan_skip=False, max_width=1000000, min_width=0,
                 skip_with_rezero=False, skip_reshape=False, **params):
        super().__init__()
        self.skip_connections = skip_connections
        filter_t, pad_t = stride_t * 2, stride_t // 2
        curr_width = width
        blocks = []
        res_scale = (1.0 if not res_scale else 1.0 / math.sqrt(depth)) if isinstance(res_scale, bool) else res_scale
        if down_t > 0:
            for i in range(down_t):
                next_width = clamp(width * channel_increase ** i, min_width, max_width)
                with_self_attn = self_attn_from is not None and self_attn_from <= i + 1
                downsample = stride_t ** (i + 1)
                in_width = input_emb_width if i == 0 else curr_width
                shape_block = nn.Conv1d(in_width, next_width, filter_t, stride_t, pad_t)
                block = [
                    BigGanSkip(shape_block, in_width, next_width, downsample=stride_t, res_scale=res_scale,
                               with_rezero=skip_with_rezero, try_match=skip_reshape) if biggan_skip else shape_block,
                    Resnet1D(next_width, depth, res_scale=res_scale, return_skip=skip_connections, norm_type=norm_type,
                             num_groups=num_groups, rezero=rezero, condition_size=condition_size,
                             with_self_attn=with_self_attn, downsample=downsample, **params),
                ]
                blocks.append(nn.Sequential(*block) if condition_size is None else
                              SkipConnectionsEncoder(block, [False, True], pass_skips=True, pass_conds=[False, True]))
                curr_width = next_width
            block = nn.Conv1d(curr_width, output_emb_width, 3, 1, 1)
            blocks.append(block)
        self.last_emb_width = width
        self.encode_block = nn.Sequential(*blocks) if not self.skip_connections else \
            SkipConnectionsEncoder(blocks, [True] * down_t + [False], pass_skips=True,
                                   pass_conds=[True] * down_t + [False])

    def forward(self, x, cond=None):
        if cond is not None:
            return self.encode_block(x, cond=cond)
        return self.encode_block(x)


class DecoderConvBock(nn.Module):
    def __init__(self, output_emb_width, down_t, stride_t, skip_connections, width, depth, res_scale=False,
                 reverse_decoder_dilation=False, rezero=False, channel_increase=1, condition_size=None,
                 self_attn_from=None, max_width=1000000, min_width=0, rezero_in_attn=False, biggan_skip=False,
                 last_emb_fixed=True, skip_with_rezero=False, skip_reshape=False, **params):
        super().__init__()
        self.last_width = clamp(width, min_width, max_width) if last_emb_fixed else output_emb_width
        self.skip_connections = skip_connections
        blocks = []
        res_scale = (1.0 if not res_scale else 1.0 / math.sqrt(depth)) if isinstance(res_scale, bool) else res_scale
        if down_t > 0:
            filter_t, pad_t = stride_t * 2, stride_t // 2
            curr_width = clamp(width * channel_increase ** (down_t - 1), min_width, max_width)
            block = nn.Conv1d(output_emb_width, curr_width, 3, 1, 1)
            blocks.append(block)
            for i in range(down_t):
                curr_width = clamp(width * channel_increase ** (down_t - i - 1), min_width, max_width)
                next_width = width * channel_increase ** (down_t - i - 2) if i < down_t - 1 else self.last_width
                next_width = clamp(next_width, min_width, max_width)
                with_self_attn = self_attn_from is not None and self_attn_from <= down_t - i
                downsample = stride_t ** (down_t - i)

                shape_block = nn.ConvTranspose1d(curr_width, next_width, filter_t, stride_t, pad_t)
                block = [
                    Resnet1D(curr_width, depth, get_skip=skip_connections, res_scale=res_scale, downsample=downsample,
                             reverse_dilation=reverse_decoder_dilation, rezero=rezero, with_self_attn=with_self_attn,
                             condition_size=condition_size, rezero_in_attn=rezero_in_attn, **params),
                    BigGanSkip(shape_block, curr_width, next_width, upsample=stride_t, res_scale=res_scale,
                               try_match=skip_reshape, with_rezero=skip_with_rezero) if biggan_skip else shape_block,
                ]
                blocks.append(nn.Sequential(*block) if condition_size is None else
                              SkipConnectionsDecoder(block, [True, False], pass_conds=[True, False]))
        self.decoder_block = nn.Sequential(*blocks) if not self.skip_connections else \
            SkipConnectionsDecoder(blocks, [False] + [True] * down_t, pass_conds=[False] + [True] * down_t)

    def forward(self, x, skips=None, cond=None):
        args = (x, skips) if skips is not None else x
        if cond is not None:
            return self.decoder_block(args, cond=cond)
        return self.decoder_block(args)


class Encoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, levels, downs_t, strides_t, skip_connections,
                 **block_kwargs):
        super().__init__()
        self.skip_connections = skip_connections
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
                             output_emb_width, down_t, stride_t, self.skip_connections, **block_kwargs_copy)
            for level, down_t, stride_t in zip(range(self.levels), downs_t, strides_t)])

    def forward(self, x, last=False, cond=None):
        N, T = x.shape[0], x.shape[-1]
        emb = self.input_emb_width
        assert_shape(x, (N, emb, T))
        xs = []

        # 64, 32, ...
        iterator = zip(range(self.levels), self.downs_t, self.strides_t)
        skips = []
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x, cond=cond)
            x, skip = (x, []) if not self.skip_connections else x
            skips.append(skip)
            emb, T = self.output_emb_width, T // (stride_t ** down_t)
            assert_shape(x, (N, emb, T))
            xs.append(x)
        xs = xs[-1] if last else xs
        return (xs, skips) if self.skip_connections else xs


class Decoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, levels, downs_t, strides_t, skip_connections,
                 **block_kwargs):
        super().__init__()
        self.skip_connections = skip_connections
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels
        self.downs_t = downs_t
        self.strides_t = strides_t

        self.level_blocks = nn.ModuleList(
            DecoderConvBock(output_emb_width, down_t, stride_t, self.skip_connections, **block_kwargs)
            for level, down_t, stride_t in zip(range(self.levels), downs_t, strides_t))

        self.out = nn.Conv1d(self.level_blocks[-1].last_width, input_emb_width, 3, 1, 1)

    def forward(self, xs, all_levels=True, skips=None, cond=None):
        assert len(xs) == (self.levels if all_levels else 1)
        skips = itertools.repeat(None) if skips is None else skips

        x = xs[-1]
        N, T = x.shape[0], x.shape[-1]
        emb = self.output_emb_width
        assert_shape(x, (N, emb, T))

        # 32, 64 ...
        iterator = reversed(list(zip(range(self.levels), self.downs_t, self.strides_t, skips)))
        for level, down_t, stride_t, skip in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x, skips=skip, cond=cond)
            emb, T = self.level_blocks[-1].last_width, T * (stride_t ** down_t)
            assert_shape(x, (N, emb, T))
            if level != 0 and all_levels:
                x = x + xs[level - 1]

        x = self.out(x)
        return x
