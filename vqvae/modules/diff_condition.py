import torch
from torch import nn as nn

from optimization.positional_encoding import get_pos_emb


class DiffusionConditioning(nn.Module):
    def __init__(self, t_cond_size, style_cond_size, listens_cond_size, pos_cond_size, time_cond_size,
                 artists_cond_size, t_enc_type, pos_enc_type, time_enc_type, listens_enc_type, listens_logarithm,
                 genre_names, noise_steps, cls_free_guidance, drop_guidance_prob, cfg_guid_weight):
        super().__init__()
        self.listens_logarithm = listens_logarithm
        self.noise_steps = noise_steps
        self.genre_names = genre_names
        self.style_embedding_size = len(genre_names)
        self.t_cond_size = t_cond_size
        self.pos_cond_size = pos_cond_size
        self.style_cond_size = style_cond_size
        self.time_cond_size = time_cond_size
        self.listens_cond_size = listens_cond_size
        self.artists_cond_size = artists_cond_size
        self.cls_free_guidance = cls_free_guidance
        self.drop_guidance_prob = drop_guidance_prob
        self.cfg_guid_weight = cfg_guid_weight

        self.use_pos_embedding = self.pos_cond_size > 0
        self.use_style_embedding = self.style_cond_size > 0
        self.use_time_embedding = self.time_cond_size > 0
        self.use_listens_embedding = self.listens_cond_size > 0
        self.use_artist_embedding = self.artists_cond_size > 0
        self.cond_with_time = self.use_time_embedding or self.use_pos_embedding

        conds = [t_cond_size, pos_cond_size, style_cond_size, time_cond_size, listens_cond_size, artists_cond_size]
        self.cond_size = sum(s for s in conds if s is not None)

        self.t_encoding = get_pos_emb(t_enc_type, token_dim=t_cond_size,
                                      n_ctx=self.noise_steps + 1, max_len=self.noise_steps + 1)[0]
        if self.use_pos_embedding:
            self.pos_enc = get_pos_emb(pos_enc_type, token_dim=pos_cond_size)[0]
        if self.use_style_embedding:
            self.style_enc = nn.Embedding(self.style_embedding_size, self.style_cond_size)
        if self.use_time_embedding:
            self.time_enc = get_pos_emb(time_enc_type, token_dim=time_cond_size, max_len=10000)[0]
        if self.use_listens_embedding:
            self.listens_enc = get_pos_emb(listens_enc_type, token_dim=listens_cond_size)[0]
        if self.use_artist_embedding:
            raise Exception("Not Implemented")

    def resolve_cfg_mask(self, drop_cond, t):
        if not self.cls_free_guidance:
            return None
        if drop_cond is not None:
            return 0 if drop_cond else 1
        probs = torch.ones_like(t) * (1 - self.drop_guidance_prob)
        mask = torch.bernoulli(probs)
        return mask.reshape(len(t), 1, 1).to(t.device)

    # drop_cond resolved according to resolve_drop_cond
    def get_conditioning(self, t, length, time_cond=None, context_cond=None, drop_cond=None):
        cfg_mask = self.resolve_cfg_mask(drop_cond, t)
        cond = self.t_encoding.forward(length=1, offset=t).unsqueeze(-1)
        if self.use_style_embedding:
            assert context_cond is not None  # sum of embeddings of all genres
            styles = [self.style_enc(torch.stack(context.genres)).sum(dim=0) for context in context_cond]
            styles = torch.stack(styles).permute(0, 2, 1)
            if cfg_mask is not None:
                styles *= cfg_mask
            cond = torch.cat([cond, styles], dim=1)
        if self.use_listens_embedding:
            listens = torch.cat([context.listens for context in context_cond])
            listens = (torch.log(listens)*10).type(torch.LongTensor) if self.listens_logarithm else listens
            listens_enc = self.listens_enc.forward(length=1, offset=listens).unsqueeze(-1)
            if cfg_mask is not None:
                listens_enc *= cfg_mask
            cond = torch.cat([cond, listens_enc], dim=1)

        if self.use_pos_embedding or self.use_time_embedding:
            cond = cond.repeat(1, 1, length)
        if self.use_pos_embedding:
            pos_enc = self.pos_enc.forward(length=length).T.unsqueeze(0).repeat(len(t), 1, 1)
            cond = torch.cat([cond, pos_enc], dim=1)
        if self.use_time_embedding:
            starts = (torch.cat([t.start / t.total for t in time_cond]) * 10000).type(torch.LongTensor)
            ends = (torch.cat([t.end / t.total for t in time_cond]) * 10000).type(torch.LongTensor)
            alls = torch.cat([torch.linspace(s, e, length, device=s.device) for s, e in zip(starts, ends)]).type(torch.LongTensor)
            time_enc = self.time_enc.forward(length=1, offset=alls).reshape(len(t), length, self.time_cond_size)
            time_enc = time_enc.permute(0, 2, 1)
            cond = torch.cat([cond, time_enc], dim=1)
        if self.use_artist_embedding:
            raise Exception("Not Implemented")
        return cond
