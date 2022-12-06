import torch
import torch as t
import torch.nn as nn
from functools import partial
from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.feature_maps import Favor
from fast_transformers.masking import TriangularCausalMask


class FastTransformer(nn.Module):
    def __init__(self,  dim, depth, heads, dim_head, feature_redraw_interval, ff_mult, feature_map_dims):
        super().__init__()
        self.transformer = TransformerEncoderBuilder.from_kwargs(
            n_layers=depth,
            n_heads=heads,
            query_dimensions=dim//heads,
            value_dimensions=dim//heads,
            feed_forward_dimensions=dim*ff_mult,
            attention_type="causal-linear",
            feature_map=partial(Favor, n_dims=feature_map_dims, redraw=feature_redraw_interval)
        ).get()

    def forward(self, x) -> torch.Tensor:
        attn_mask = TriangularCausalMask(x.shape[1], device=x.device.type)
        return self.transformer(x, attn_mask=attn_mask)
