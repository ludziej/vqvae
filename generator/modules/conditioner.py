from typing import NamedTuple, Optional

import torch.nn as nn
import torch
from generator.modules.uptokenconditioner import UpTokenConditioner
from generator.modules.positional_encoding import TrainablePositionalEncoding, FourierFeaturesPositionalEncoding,\
    BPMPositionalEncoding


class GenerationParams(NamedTuple):
    artist: Optional[torch.Tensor] = None
    time: Optional[torch.Tensor] = None
    bpm: Optional[torch.Tensor] = None
    bpm_offset: Optional[torch.Tensor] = None
    # here add also BPM/genre etc


class Conditioner(nn.Module):
    def __init__(self, preprocessing, conditioning_dropout, conds_kwargs, z_shapes, bins, token_dim, level,
                 bins_init_scale, pos_enc_type, init_bins_from_vqvae, pos_enc_lvl_over_bit, context_on_level,
                 conditioning_concat, n_ctx, pos_init_scale):
        super().__init__()
        self.n_ctx = n_ctx
        self.pos_init_scale = pos_init_scale
        self.pos_enc_lvl_over_bit = pos_enc_lvl_over_bit
        self.init_bins_from_vqvae = init_bins_from_vqvae
        self.context_on_level = context_on_level
        self.pos_enc_type = pos_enc_type
        self.level = level
        self.bins_init_scale = bins_init_scale
        self.token_dim = token_dim
        self.bins = bins
        self.conditioning_concat = conditioning_concat

        # is_absolute needs to recalculate when we move windows by 1
        self.pos_emb, self.pos_embeddings_is_absolute = self.create_pos_emb()
        self.cond_dropout = nn.Dropout(conditioning_dropout)
        self.x_emb = nn.Embedding(self.bins, self.token_dim)
        self.init_emb(self.x_emb)
        if self.context_on_level:
            u_level = self.level + 1
            bins_init = None if not self.init_bins_from_vqvae else self.get_vqvae_bins(preprocessing, u_level)
            self.conditioner = UpTokenConditioner(input_shape=z_shapes[u_level], bins=preprocessing.l_bins,
                                                  out_width=self.token_dim, down_t=preprocessing.downs_t[u_level],
                                                  bins_init=bins_init, stride_t=preprocessing.strides_t[u_level],
                                                  **conds_kwargs)

    def get_vqvae_bins(self, preprocessing, level):
        return preprocessing.generator.bottleneck.level_blocks[level].k.detach()

    def create_pos_emb(self):
        if self.pos_enc_type == "trainable":
            return TrainablePositionalEncoding(input_shape=(self.n_ctx,), width=self.token_dim,
                                               init_scale=self.pos_init_scale), False
        elif self.pos_enc_type == "fourier":
            return FourierFeaturesPositionalEncoding(depth=self.token_dim), False
        elif self.pos_enc_type == "bpm":
            return BPMPositionalEncoding(depth=self.token_dim, levels_per_bit=self.pos_enc_lvl_over_bit), False
        else:
            raise Exception(f"Unknown pos_enc_type={self.pos_enc_type}")

    def init_emb(self, bins: nn.Module):
        if self.init_bins_from_vqvae:
            bins.weight = torch.nn.Parameter((self.get_vqvae_bins(self.level)))
        else:
            nn.init.normal_(bins.weight, std=self.bins_init_scale)

    # TODO implement
    def get_context_conditioning(self, context, bs, length, device=None):
        raise NotImplementedError("currently no context conditioning")

    # TODO implement
    def get_time_conditioning(self, time, bs, length, device=None):
        raise NotImplementedError("currently no time conditioning")

    # TODO implement batched version for different lengths (like jukebox)
    def get_lvl_conditioning(self, up_level, bs, length, device=None):
        from_up_lvl = self.conditioner(up_level)
        assert from_up_lvl.shape == (bs, length, self.token_dim)
        return from_up_lvl

    def get_pos_emb(self, bs, device, length=None, bpm=None, bpm_offset=None,):
        pos_emb = self.pos_emb(length=length, bpm=bpm, bpm_offset=bpm_offset, device=device)
        pos_emb = pos_emb.unsqueeze(0).repeat(bs, 1, 1)
#        if start_at != 0:  # this does the same as applying relative conditioning, and both exclude each other
#            pad = torch.zeros((bs, start_at, self.token_dim), dtype=pos_emb.dtype, device=pos_emb.device)
#            pos_emb = torch.cat([pad, pos_emb], dim=1)
        return pos_emb

    def get_all_conditioning(self, bs, length, device, params: GenerationParams, up_tokens=None):
        args = dict(bs=bs, length=length, device=device)
        conditionings = dict(
            pos_cond=(True, lambda: self.get_pos_emb(bpm=params.bpm, bpm_offset=params.bpm_offset, **args)),
            lvl_cond=(up_tokens, lambda: self.get_lvl_conditioning(up_tokens, **args)),
            context_cond=(params.artist, lambda: self.get_context_conditioning(params.artist, **args)),
            time_cond=(params.time, lambda: self.get_time_conditioning(params.time, **args))
        )
        return {k: f() for k, (v, f) in conditionings.items() if v is not None}

    def get_conditioned_emb(self, tokens, up_tokens=None, gen_params=None):
        b, t = tokens.shape[:2]
        embeddings = self.x_emb(tokens)
        conditioning = self.get_all_conditioning(b, t, params=gen_params, up_tokens=up_tokens, device=tokens.device)
        embeddings = self.join_conditioning(embeddings, **conditioning)
        return embeddings

    # TODO implement
    def conditioning_concat_projection(self, x, conds):
        raise NotImplementedError("concat conditioning not implemented")

    def join_conditioning(self, x, lvl_cond=None, context_cond=None, pos_cond=None, time_cond=None,
                          token_interv=None):
        x = x[:, token_interv[0]: token_interv[1], :] if token_interv is not None else x
        bs, size = x.shape[:2]
        conds = self.apply_cond_relative_absolute(lvl_cond, context_cond, pos_cond, time_cond, size, token_interv)
        conds = [self.cond_dropout(c) for c in conds]
        x = self.cond_dropout(x)
        if self.conditioning_concat:
            return self.conditioning_concat_projection(x, conds)
        else:
            assert all(c.shape == x.shape for c in conds)  # each conditioning has to have same size as x
            return x + sum(conds)

    def apply_cond_relative_absolute(self, lvl_cond, context_cond, pos_cond, time_cond, size, token_interv):
        conds = [lvl_cond, context_cond, pos_cond, time_cond]
        is_cond_relative = [False, True, not self.pos_embeddings_is_absolute, False]
        # is this conditioning relative - taken according to sliding window, or absolute - according to tracks position
        conds = [(c, m) for c, m in zip(conds, is_cond_relative) if c is not None]
        conds = [c if token_interv is None else
                 c[:, :size] if moving else
                 c[:, token_interv[0]: token_interv[1]] for c, moving in conds]
        return conds

    def get_autoregressive_conditioning(self, x_emb_cond, tokens_emb, ctx_start, i, conditioning, seq_len):
        if not self.pos_embeddings_is_absolute:
            return self.join_conditioning(tokens_emb, token_interv=(ctx_start, i), **conditioning)
        else:
            if x_emb_cond is None:
                x_emb_cond = torch.zeros((tokens_emb.shape[0], seq_len, self.token_dim),
                                         device=tokens_emb.device, dtype=tokens_emb.dtype)
                x_emb_cond[:i] = self.join_conditioning(tokens_emb, token_interv=(ctx_start, i), **conditioning)
            else:
                x_emb_cond[i - 1: i] = self.join_conditioning(tokens_emb, token_interv=(i - 1, i), **conditioning)
            return x_emb_cond
