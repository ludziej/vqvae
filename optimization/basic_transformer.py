import torch.nn as nn
from optimization.layers import ReZero, _get_clones, Residual


class SelfAttentionBlock(nn.Module):
    def __init__(self, width, heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        return self.self_attn(x, x, x, attn_mask=None, key_padding_mask=None, need_weights=False)[0]


class TransformerLayer(nn.Module):
    def __init__(self, width, heads, dim_ff=2, dropout=0.1, prenorm=False, layer_norm_eps=1e-5, rezero=False):
        super().__init__()
        self.norm_first = prenorm

        self_attn = SelfAttentionBlock(width, heads, dropout=dropout)
        linear1 = nn.Linear(width, dim_ff)
        linear2 = nn.Linear(dim_ff, width)
        dropout_fn = nn.Dropout(dropout) if dropout > 0 else nn.Sequential()
        activation = nn.ReLU()

        sa_block = nn.Sequential(self_attn, dropout_fn)
        fc_block = nn.Sequential(linear1, activation, dropout_fn, linear2, dropout_fn)

        if rezero:
            self.sa_block = ReZero(sa_block)
            self.fc_block = ReZero(fc_block)
        else:
            norm1 = nn.LayerNorm(width, eps=layer_norm_eps)
            norm2 = nn.LayerNorm(width, eps=layer_norm_eps)
            if prenorm:
                self.sa_block = Residual(nn.Sequential(sa_block, norm1))
                self.fc_block = Residual(nn.Sequential(fc_block, norm2))
            else:
                self.sa_block = nn.Sequential(Residual(sa_block), norm1)
                self.fc_block = nn.Sequential(Residual(fc_block), norm2)

    def forward(self, x):
        x = self.sa_block(x)
        x = self.fc_block(x)
        return x


class TransformerEncoderStack(nn.Module):
    def __init__(self, layer, depth):
        super().__init__()
        self.layers = nn.Sequential(*_get_clones(layer, depth))

    def forward(self, x):
        return self.layers(x)


class TransformerEncoder(nn.Module):
    def __init__(self, depth, heads, width, dropout, ff_mul, prenorm=False, rezero=False):
        super().__init__()
        self.depth = depth
        self.heads = heads
        transformer_layer = TransformerLayer(width, heads, width * ff_mul, dropout, prenorm=prenorm, rezero=rezero)
        self.transformer = TransformerEncoderStack(transformer_layer, depth)

    def forward(self, xs):
        xs = xs.permute(0, 2, 1)
        xs = self.transformer(xs)
        return xs.permute(0, 2, 1)
