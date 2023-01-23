import itertools

from torch import nn as nn
import torch as t

from utils.misc import exists, default
from utils.old_ml_utils.misc import assert_shape
from vqvae.modules.quant_bottleneck import Bottleneck, NoBottleneck
from vqvae.modules.vae import VAEBottleneck
from vqvae.modules.encdec import Encoder, Decoder
from vqvae.modules.audio_logger import AudioLogger


class WavAutoEncoder(nn.Module):
    def __init__(self, sr, downs_t, emb_width, input_channels, l_bins, levels, mu,
                 norm_before_vqvae, strides_t, bottleneck_type, skip_connections, multipliers,
                 fixed_commit=False,
                 log_weights_norm=False, base_model=None, block_params=None, **params):
        super().__init__()
        block_params = default(block_params, params)
        self.sr = sr
        self.emb_width = emb_width
        self.multipliers = multipliers
        self.skip_connections = skip_connections
        self.input_channels = input_channels
        self.levels = levels
        self.block_params = block_params
        assert len(multipliers) == levels, "Invalid number of multipliers"

        model = default(base_model, self)
        self.audio_logger = AudioLogger(sr=self.sr, model=model, use_weights_logging=log_weights_norm)
        encoders = nn.ModuleList([Encoder(input_channels, emb_width, level + 1, downs_t[:level + 1],
                                          strides_t[:level + 1], self.skip_connections, **self.block_kwargs(level))
                                  for level in range(levels)])
        decoders = nn.ModuleList([Decoder(input_channels, emb_width, level + 1, downs_t[:level + 1],
                                          strides_t[:level + 1], self.skip_connections, **self.block_kwargs(level))
                                  for level in range(levels)])
        bottleneck, self.name = self.get_bottleneck(bottleneck_type, l_bins, emb_width, mu,
                                                    levels, norm_before_vqvae, fixed_commit)
        self.encoders = encoders
        self.decoders = decoders
        self.bottleneck = bottleneck

    def block_kwargs(self, level, multiply=True):
        this_block_kwargs = dict(self.block_params)
        if multiply:
            this_block_kwargs["width"] *= self.multipliers[level]
        this_block_kwargs["depth"] *= self.multipliers[level]
        return this_block_kwargs

    def get_bottleneck(self, bottleneck_type, l_bins, emb_width, mu, levels, norm_before_vqvae, fixed_commit):
        if bottleneck_type == "vqvae":
            return Bottleneck(l_bins, emb_width, mu, levels, norm_before_vqvae, fixed_commit), "VQ-VAE"
        elif bottleneck_type == "none" and self.skip_connections:
            return NoBottleneck(levels), "U-Net"
        elif bottleneck_type == "none":
            return NoBottleneck(levels), "Auto Encoder"
        elif bottleneck_type == "vae":
            return VAEBottleneck(emb_width, levels), "VAE"
        else:
            raise Exception(f"Unknown bottleneck type {bottleneck_type}")

    def forward(self, x_in, cond=None):
        encode_data = [encoder(x_in, last=True) for encoder in self.encoders]
        x_encoded, x_skips = zip(*encode_data) if self.skip_connections else (encode_data, itertools.repeat(None))

        bottleneck_data = self.bottleneck(x_encoded) if cond is None else self.bottleneck(x_encoded, cond)
        zs, xs_quantised, bottleneck_losses, prenorms, metrics = bottleneck_data

        x_outs = [decoder(xs_quantised[level:level+1], all_levels=False, skips=skip)
                  for level, (decoder, skip) in enumerate(zip(self.decoders, x_skips))]
        [assert_shape(x_out, x_in.shape) for x_out in x_outs]
        return x_outs, bottleneck_losses, prenorms, metrics

    def preprocess(self, x):
        # x: NTC [-1,1] -> NCT [-1,1]
        assert len(x.shape) == 3
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # x: NTC [-1,1] <- NCT [-1,1]
        x = x.permute(0, 2, 1)
        return x

    def decode_one_chunk(self, zs, start_level=0, end_level=None):
        # Decode
        if end_level is None:
            end_level = self.levels
        assert len(zs) == end_level - start_level
        xs_quantised = self.bottleneck.decode(zs, start_level=start_level, end_level=end_level)
        assert len(xs_quantised) == end_level - start_level

        # Use only lowest level
        decoder, x_quantised = self.decoders[start_level], xs_quantised[0:1]
        x_out = decoder(x_quantised, all_levels=False)
        x_out = self.postprocess(x_out)
        return x_out

    def encode_one_chunk(self, x, start_level=0, end_level=None):
        # Encode
        if end_level is None:
            end_level = self.levels
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])
        zs = self.bottleneck.encode(xs)
        return zs[start_level:end_level]

    def decode(self, zs, start_level=0, end_level=None, bs_chunks=1):
        z_chunks = [t.chunk(z, bs_chunks, dim=0) for z in zs]
        x_outs = []
        for i in range(bs_chunks):
            zs_i = [z_chunk[i] for z_chunk in z_chunks]
            x_out = self.decode_one_chunk(zs_i, start_level=start_level, end_level=end_level)
            x_outs.append(x_out)
        return t.cat(x_outs, dim=0)

    def encode(self, x, start_level=0, end_level=None, bs_chunks=1):
        x_chunks = t.chunk(x, bs_chunks, dim=0)
        zs_list = []
        for x_i in x_chunks:
            zs_i = self.encode_one_chunk(x_i, start_level=start_level, end_level=end_level)
            zs_list.append(zs_i)
        zs = [t.cat(zs_level_list, dim=0) for zs_level_list in zip(*zs_list)]
        return zs
