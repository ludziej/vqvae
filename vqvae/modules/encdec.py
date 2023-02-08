import itertools

import torch.nn as nn
from vqvae.modules.resnet import Resnet1D
from utils.old_ml_utils.misc import assert_shape
from vqvae.modules.skip_connections import SkipConnectionsDecoder, SkipConnectionsEncoder


class EncoderConvBlock(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t, stride_t, skip_connections, width, depth, m_conv,
                 norm_type, dilation_growth_rate=1, dilation_cycle=None, res_scale=False, leaky_param=1e-2,
                 use_weight_standard=True, num_groups=32, use_bias=False, concat_skip=False, rezero=False,
                 skip_connections_step=1, channel_increase=1, condition_size=None, self_attn_from=None,
                 cond_with_time=False, **params):
        super().__init__()
        self.skip_connections = skip_connections
        filter_t, pad_t = stride_t * 2, stride_t // 2
        curr_width = width
        blocks = []
        if down_t > 0:
            for i in range(down_t):
                next_width = width * channel_increase ** i
                with_self_attn = self_attn_from is not None and self_attn_from <= i + 1
                downsample = stride_t ** (i + 1)
                block = [
                    nn.Conv1d(input_emb_width if i == 0 else curr_width, next_width, filter_t, stride_t, pad_t),
                    Resnet1D(next_width, depth, m_conv, dilation_growth_rate, dilation_cycle, res_scale,
                             return_skip=skip_connections, norm_type=norm_type, leaky_param=leaky_param,
                             use_weight_standard=use_weight_standard, num_groups=num_groups, use_bias=use_bias,
                             concat_skip=concat_skip, skip_connections_step=skip_connections_step, rezero=rezero,
                             condition_size=condition_size, with_self_attn=with_self_attn,
                             cond_with_time=cond_with_time, downsample=downsample),
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
    def __init__(self, input_emb_width, output_emb_width, down_t, stride_t,
                 skip_connections, width, depth, m_conv, norm_type, dilation_growth_rate=1, dilation_cycle=None,
                 res_scale=False, reverse_decoder_dilation=False, leaky_param=1e-2, use_weight_standard=True,
                 num_groups=32, use_bias=False, concat_skip=False, rezero=False, skip_connections_step=1,
                 channel_increase=1, condition_size=None, self_attn_from=None, cond_with_time=False, **params):
        super().__init__()
        self.skip_connections = skip_connections
        blocks = []
        if down_t > 0:
            filter_t, pad_t = stride_t * 2, stride_t // 2
            block = nn.Conv1d(output_emb_width, width * channel_increase ** (down_t - 1), 3, 1, 1)
            blocks.append(block)
            for i in range(down_t):
                curr_width = width * channel_increase ** (down_t - i - 1)
                next_width = input_emb_width if i == (down_t - 1) else width * channel_increase ** (down_t - i - 2)
                with_self_attn = self_attn_from is not None and self_attn_from <= down_t - i
                downsample = stride_t ** (down_t - i)
                block = [
                    Resnet1D(curr_width, depth, m_conv, dilation_growth_rate, dilation_cycle, leaky_param=leaky_param,
                             get_skip=skip_connections, norm_type=norm_type, res_scale=res_scale, num_groups=num_groups,
                             reverse_dilation=reverse_decoder_dilation, use_weight_standard=use_weight_standard,
                             use_bias=use_bias, concat_skip=concat_skip, rezero=rezero, with_self_attn=with_self_attn,
                             skip_connections_step=skip_connections_step, condition_size=condition_size,
                             cond_with_time=cond_with_time, downsample=downsample),
                    nn.ConvTranspose1d(curr_width, next_width, filter_t, stride_t, pad_t),
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
            DecoderConvBock(output_emb_width, output_emb_width, down_t, stride_t, self.skip_connections, **block_kwargs)
            for level, down_t, stride_t in zip(range(self.levels), downs_t, strides_t))

        self.out = nn.Conv1d(output_emb_width, input_emb_width, 3, 1, 1)

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
            emb, T = self.output_emb_width, T * (stride_t ** down_t)
            assert_shape(x, (N, emb, T))
            if level != 0 and all_levels:
                x = x + xs[level - 1]

        x = self.out(x)
        return x
