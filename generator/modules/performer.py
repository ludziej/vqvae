import torch
import torch.nn.functional as F
from torch import nn

from functools import partial

from performer_pytorch import SelfAttention, CrossAttention, ProjectionUpdater
from performer_pytorch.reversible import ReversibleSequence, SequentialSequence
from performer_pytorch.autoregressive_wrapper import top_k, repetition_penalty_fn
from utils.misc import default, exists


def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val


def empty(tensor):
    return tensor.numel() == 0


class ReZero(nn.Module):
    def __init__(self, fn, init=0.):  # originally init was 1e-3
        super().__init__()
        self.g = nn.Parameter(torch.tensor(init))
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.g


class PreScaleNorm(nn.Module):
    def __init__(self, dim, fn, eps=1e-5):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x, **kwargs):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        x = x / n * self.g
        return self.fn(x, **kwargs)


class PreLayerNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim=-1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim=self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim=self.dim)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0., activation=None, glu=False):
        super().__init__()
        activation = default(activation, nn.GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x


def shift(t, amount, mask=None):
    if amount == 0:
        return t

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)

    return F.pad(t, (0, 0, amount, -amount), value=0.)


class PreShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get('mask', None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim=-1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(map(lambda args: shift(*args, mask=mask), zip(segments_to_shift, shifts)))
        x = torch.cat((*segments_to_shift, *rest), dim=-1)
        return self.fn(x, **kwargs)


class Performer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, local_attn_heads=0, local_window_size=256, causal=False,
                 ff_mult=4, nb_features=None, feature_redraw_interval=1000, reversible=False, ff_chunks=1,
                 generalized_attention=False, kernel_fn=nn.ReLU(), use_scalenorm=False, use_rezero=False,
                 ff_glu=False, ff_dropout=0., attn_dropout=0., cross_attend=False, no_projection=False,
                 auto_check_redraw=True, qkv_bias=True, attn_out_bias=True, shift_tokens=False, rezero_init=1e-3):
        super().__init__()
        layers = nn.ModuleList([])
        local_attn_heads = cast_tuple(local_attn_heads)
        local_attn_heads = local_attn_heads * depth if len(local_attn_heads) == 1 else local_attn_heads
        assert len(
            local_attn_heads) == depth, 'tuple specifying number of local attention heads per depth must be equal to the total depth'
        assert all(map(lambda n: n >= 0 and n <= heads,
                       local_attn_heads)), 'local attention head value must be less than the total number of heads'

        if use_scalenorm:
            wrapper_fn = partial(PreScaleNorm, dim)
        elif use_rezero:
            wrapper_fn = partial(ReZero, init=rezero_init)
        else:
            wrapper_fn = partial(PreLayerNorm, dim)

        for _, local_heads in zip(range(depth), local_attn_heads):

            attn = SelfAttention(dim, causal=causal, heads=heads, dim_head=dim_head, local_heads=local_heads,
                                 local_window_size=local_window_size, nb_features=nb_features,
                                 generalized_attention=generalized_attention, kernel_fn=kernel_fn, dropout=attn_dropout,
                                 no_projection=no_projection, qkv_bias=qkv_bias, attn_out_bias=attn_out_bias)
            ff = Chunk(ff_chunks, FeedForward(dim, mult=ff_mult, dropout=ff_dropout, glu=ff_glu), along_dim=1)

            if shift_tokens:
                shift = (0, 1) if causal else (-1, 0, 1)
                attn, ff = map(lambda t: PreShiftTokens(shift, t), (attn, ff))

            attn, ff = map(wrapper_fn, (attn, ff))
            layers.append(nn.ModuleList([attn, ff]))

            if not cross_attend:
                continue

            layers.append(nn.ModuleList([
                wrapper_fn(CrossAttention(dim, heads=heads, dim_head=dim_head, nb_features=nb_features,
                                          generalized_attention=generalized_attention, kernel_fn=kernel_fn,
                                          dropout=attn_dropout, no_projection=no_projection, qkv_bias=qkv_bias,
                                          attn_out_bias=attn_out_bias)),
                wrapper_fn(
                    Chunk(ff_chunks, FeedForward(dim, mult=ff_mult, dropout=ff_dropout, glu=ff_glu), along_dim=1))
            ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence

        route_attn = ((True, False),) * depth * (2 if cross_attend else 1)
        route_context = ((False, False), (True, False)) * depth
        attn_route_map = {'mask': route_attn, 'pos_emb': route_attn}
        context_route_map = {'context': route_context, 'context_mask': route_context} if cross_attend else {}
        self.net = execute_type(layers, args_route={**attn_route_map, **context_route_map})

        # keeping track of when to redraw projections for all attention layers
        self.auto_check_redraw = auto_check_redraw
        self.proj_updater = ProjectionUpdater(self.net, feature_redraw_interval)

    def fix_projection_matrices_(self):
        self.proj_updater.feature_redraw_interval = None

    def forward(self, x, **kwargs):
        if self.auto_check_redraw:
            self.proj_updater.redraw_projections()
        return self.net(x, **kwargs)
