import torch.nn as nn
from utils.old_ml_utils.misc import assert_shape
from vqvae.modules.encdec import DecoderConvBock
from optimization.normalization import CustomNormalization


# conditions on one level above
class UpTokenConditioner(nn.Module):
    def __init__(self, input_shape, bins, down_t, stride_t, out_width, init_scale, zero_out, res_scale, norm_type,
                 bins_init=None, **block_kwargs):
        super().__init__()
        self.x_shape = input_shape

        # Embedding
        self.width = out_width
        self.x_emb = nn.Embedding(bins, out_width)
        if bins_init is None:
            nn.init.normal_(self.x_emb.weight, std=0.02 * init_scale)
        else:
            self.x_emb.weight = bins_init

        # Conditioner
        self.cond = DecoderConvBock(self.width, self.width, down_t, stride_t, zero_out=zero_out, res_scale=res_scale,
                                    norm_type=norm_type, **block_kwargs)
        self.ln = CustomNormalization(self.width, norm_type=norm_type)

    def preprocess(self, x):
        x = x.permute(0,2,1) # NTC -> NCT
        return x

    def postprocess(self, x):
        x = x.permute(0,2,1) # NCT -> NTC
        return x

    def forward(self, x, x_cond=None):
        N = x.shape[0]
        assert_shape(x, (N, *self.x_shape))
        if x_cond is not None:
            assert_shape(x_cond, (N, *self.x_shape, self.width))
        else:
            x_cond = 0.0
        # Embed x
        x = x.long()
        x = self.x_emb(x)
        assert_shape(x, (N, *self.x_shape, self.width))
        x = x + x_cond

        # Run conditioner
        x = self.preprocess(x)
        x = self.cond(x)
        x = self.postprocess(x)
        x = self.ln(x)
        return x
