from torch import nn as nn

from vqvae.modules.quant_bottleneck import Bottleneck, NoBottleneck
from vqvae.modules.vae import VAEBottleneck
from vqvae.modules.encdec import Encoder, Decoder


class VQVAEGenerator(nn.Module):
    def __init__(self, _block_kwargs, downs_t, emb_width, fixed_commit, input_channels, l_bins, levels, mu,
                 norm_before_vqvae, strides_t, bottleneck_type):
        super().__init__()
        self.levels = levels

        encoders = nn.ModuleList([Encoder(input_channels, emb_width, level + 1,
                                          downs_t[:level + 1], strides_t[:level + 1], **_block_kwargs(level))
                                  for level in range(levels)])
        decoders = nn.ModuleList([Decoder(input_channels, emb_width, level + 1,
                                          downs_t[:level + 1], strides_t[:level + 1], **_block_kwargs(level))
                                  for level in range(levels)])
        bottleneck = self.get_bottleneck(bottleneck_type, l_bins, emb_width, mu,
                                         levels, norm_before_vqvae, fixed_commit)
        self.encoders = encoders
        self.decoders = decoders
        self.bottleneck = bottleneck

    def get_bottleneck(self, bottleneck_type, l_bins, emb_width, mu, levels, norm_before_vqvae, fixed_commit):
        if bottleneck_type == "vqvae":
            return Bottleneck(l_bins, emb_width, mu, levels, norm_before_vqvae, fixed_commit)
        elif bottleneck_type == "none":
            return NoBottleneck(levels)
        elif bottleneck_type == "vae":
            return VAEBottleneck(emb_width, levels)
        else:
            raise Exception(f"Unknown bottleneck type {bottleneck_type}")

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
