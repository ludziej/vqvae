import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from performer_pytorch import Performer
from vqvae.model import VQVAE
from generator.conditioner import Conditioner, PositionEmbedding
import torch.nn.functional as F
from performer_pytorch.autoregressive_wrapper import top_k, repetition_penalty_fn


class LevelGenerator(LightningModule):
    def __init__(self, vqvae: VQVAE, level: int, log_sample_size: int, context_on_level: int,
                 dim: int, depth: int, heads: int,  lr: float, start_gen_sample_len: int,
                 log_starting_context_perc: int, log_context_time: float, n_ctx: int,
                 pos_init_scale: int, bins_init_scale: float, dim_head: int,
                 conds_kwargs: dict, init_bins_from_vqvae: bool, layer_for_logits: bool, **params):
        super().__init__()
        self.n_ctx = n_ctx
        self.level = level
        self.log_sample_bs, self.log_sample_size = log_sample_size
        self.preprocessing = vqvae
        self.preprocessing.freeze()
        self.sr = vqvae.sr
        self.start_gen_sample_len = start_gen_sample_len
        self.lr = lr
        self.dim = dim
        self.max_seq_len = n_ctx
        self.context_on_level = context_on_level
        self.log_starting_context_perc = log_starting_context_perc
        self.log_context_time = log_context_time
        self.pos_init_scale = pos_init_scale
        self.bins_init_scale = bins_init_scale
        self.bins = self.preprocessing.l_bins
        self.eos_token = None
        self.conditioning_concat = False
        self.init_bins_from_vqvae = init_bins_from_vqvae
        self.layer_for_logits = layer_for_logits
        self.log_nr = {"val_": 0, "": 0, "test_": 0}

        self.transformer = Performer(causal=True, dim=dim, depth=depth, heads=heads, dim_head=dim_head)
        self.pos_emb = PositionEmbedding(input_shape=(self.n_ctx,), width=self.dim, init_scale=self.pos_init_scale)
        self.pos_embeddings_is_absolute = True  # needs to recalculate when we move windows by 1
        self.x_emb = nn.Embedding(self.bins, self.dim)
        self.init_emb(self.x_emb)

        if self.layer_for_logits:
            self.final_layer_norm = nn.LayerNorm(dim)
            self.to_out = nn.Linear(dim, self.bins)

        z_shapes = [(z_shape[0] * self.n_ctx // vqvae.z_shapes[self.level][0],) for z_shape in vqvae.z_shapes]
        if self.context_on_level:
            u_level = self.level + 1
            # TODO add bins_init parameter with weights from vqvae if self.init_bins_from_vqvae
            self.conditioner = Conditioner(input_shape=z_shapes[u_level], bins=self.preprocessing.l_bins,
                                           down_t=self.preprocessing.downs_t[u_level], out_width=dim,
                                           stride_t=self.preprocessing.strides_t[u_level], **conds_kwargs)

    def init_emb(self, bins: nn.Module):
        if self.init_bins_from_vqvae:
            # TODO implement fully
            raise NotImplementedError("currently no embedding initialization from vqvae")
        else:
            nn.init.normal_(bins.weight, std=0.02 * self.bins_init_scale)

    def get_transformer_logits(self, x_emb):
        x = self.transformer(x_emb)
        if self.layer_for_logits:
            x = self.final_layer_norm(x)
            x = self.to_out(x)
            return x
        else:
            # TODO implement some version using embedding location for classification task, instead of just linear layer
            raise NotImplementedError("currently no method of token classification other then plain layer")

    def forward(self, sound: torch.Tensor, context=None, time=None) -> torch.Tensor:
        tokens, up_tokens = self.get_tokens(sound)
        embedding = self.get_conditioned_emb(tokens, up_tokens, context, time)
        loss = self.autoregressive_forward_loss(embedding, tokens)
        return loss

    def autoregressive_forward_loss(self, embedding, tokens) -> torch.Tensor:
        x_in = embedding[:, :-1]
        x_tok = tokens[:, 1:]
        out = self.get_transformer_logits(x_in)

        assert out.shape[:2] == x_in.shape[:2] and out.shape[2] == self.bins
        loss = F.cross_entropy(out.reshape(-1, self.bins), x_tok.reshape(-1))
        return loss

    def get_tokens(self, sound):
        endlevel = self.level + 1 if not self.context_on_level else self.level + 2
        tokens = [x.detach() for x in self.preprocessing.encode(sound, start_level=self.level, end_level=endlevel)]
        tokens, up_tokens = tokens if self.context_on_level else (tokens[0], None)
        return tokens, up_tokens

    # conditionals

    # TODO implement
    def get_context_conditioning(self, context, bs, length):
        raise NotImplementedError("currently no context conditioning")

    # TODO implement
    def get_time_conditioning(self, time, bs, length):
        raise NotImplementedError("currently no time conditioning")

    # TODO implement batched version for different lengths (like jukebox)
    def get_lvl_conditioning(self, up_level, bs, length):
        from_up_lvl = self.conditioner(up_level)
        assert from_up_lvl.shape == (bs, length, self.dim)
        return from_up_lvl

    def get_pos_emb(self, bs, start_at=0, length=None):
        pos_emb = self.pos_emb().repeat(bs, 1).reshape(bs, self.n_ctx, self.dim)
        if start_at != 0:
            pad = torch.zeros((bs, start_at, self.dim), dtype=pos_emb.dtype, device=pos_emb.device)
            pos_emb = torch.cat([pad, pos_emb], dim=1)
        return pos_emb

    def get_all_conditioning(self, bs, length, context=None, up_tokens=None, time=None):
        pos_cond = self.get_pos_emb(bs=bs, length=length)
        lvl_cond = self.get_lvl_conditioning(up_tokens, bs, length) if up_tokens is not None else None
        context_cond = self.get_context_conditioning(context, bs, length) if context is not None else None
        time_cond = self.get_time_conditioning(time, bs, length) if time is not None else None
        return dict(pos_cond=pos_cond, lvl_cond=lvl_cond, context_cond=context_cond, time_cond=time_cond)

    def get_conditioned_emb(self, tokens, up_tokens=None, context=None, time=None):
        b, t = tokens.shape[:2]
        embeddings = self.x_emb(tokens)
        conditioning = self.get_all_conditioning(b, t, context=context, up_tokens=up_tokens, time=time)
        embeddings = self.join_conditioning(embeddings, **conditioning)
        return embeddings

    # TODO implement
    def conditioning_concat_projection(self, x, conds):
        raise NotImplementedError("concat conditioning not implemented")

    def join_conditioning(self, x, lvl_cond=None, context_cond=None, pos_cond=None, time_cond=None,
                          token_interv=None):
        x = x[:, token_interv[0]: token_interv[1], :] if token_interv is not None else x
        bs, size = x.shape[:2]
        conds = [lvl_cond, context_cond, pos_cond, time_cond]
        is_const_window = [False, True, True, False]
        conds = [(c, m) for c, m in zip(conds, is_const_window) if c is not None]
        conds = [c if token_interv is None else c[:size] if moving else c[token_interv[0]: token_interv[1]]
                 for c, moving in conds]
        if self.conditioning_concat:
            return self.conditioning_concat_projection(x, conds)
        else:
            assert all(c.shape == x.shape for c in conds)  # each conditioning has to have same size as x
            return x + sum(conds)

    # generation

    def logits_to_sample(self, logits, prev_out, temperature=1., filter_logits_fn=top_k, filter_thres=0.9,
                 repetition_penalty=1.0, repetition_penalty_ctx=32):
        if repetition_penalty > 1.0:
            logits = repetition_penalty_fn(logits, prev_out[:, -repetition_penalty_ctx:], theta=repetition_penalty)
        filtered_logits = filter_logits_fn(logits, thres=filter_thres)
        probs = F.softmax(filtered_logits / temperature, dim=-1)
        sample = torch.multinomial(probs, 1)
        return sample

    @torch.no_grad()
    def autoregressive_generate_prior(self, start_tokens, seq_len, conditioning=None, **sampling_kwargs):
        was_training = self.transformer.training
        b, t = start_tokens.shape
        self.transformer.eval()
        tokens = start_tokens
        tokens_emb = self.x_emb(start_tokens)

        for i in range(t, seq_len):
            ctx_start = max(i - self.n_ctx + 1, 0)

            # TODO all this conditioning can be done once per element, if positional_embedding is not absolute
            x_emb_cond = self.join_conditioning(tokens_emb, token_interv=(ctx_start, i), **conditioning)

            logits = self.get_transformer_logits(x_emb_cond)[:, -1, :]
            sample_token = self.logits_to_sample(logits, **sampling_kwargs)
            sample_emb = self.x_emb(sample_token)

            tokens = torch.cat((tokens, sample_token), dim=1)
            tokens_emb = torch.cat((tokens_emb, sample_emb), dim=1)
            if self.eos_token is not None and (sample_token == self.eos_token).all():
                break

        tokens = tokens[:, t:]
        self.transformer.train(was_training)
        return tokens

    def decode_sound(self, tokens):
        return self.preprocessing.decode([tokens], start_level=self.level, end_level=self.level + 1).squeeze(2)

    def recreate_beginning(self, size, bs=1):
        return torch.randint(size, (bs, self.start_gen_sample_len), device=self.device)

    @torch.no_grad()
    def generate_no_sound(self, seq_len: int, start_random_size=10, bs=1, context=None, time=None, **sampling_kwargs):
        beginning = self.recreate_beginning(start_random_size, bs)
        conditioning = self.get_all_conditionings(bs, seq_len, context=context, time=time)
        out_tokens = self.autoregressive_generate_prior(beginning, seq_len, conditioning, sampling_kwargs)
        sound = self.decode_sound(out_tokens)
        return sound

    @torch.no_grad()
    def generate_from_sound(self, sound: torch.Tensor, prefix_token_perc: float, seq_len=None, context=None,
                            time=None, with_begin=True, **sampling_kwargs):
        tokens, up_tokens = self.get_tokens(sound)
        bs, t_len = tokens.shape[:2]
        seq_len = seq_len if seq_len is not None else t_len
        beginning = tokens[:int(t_len * prefix_token_perc)]
        conditioning = self.get_all_conditioning(bs, seq_len, context=context, up_tokens=up_tokens, time=time)

        out_tokens = self.autoregressive_generate_prior(beginning, seq_len, conditioning, **sampling_kwargs)
        out_tokens = torch.cat([beginning, out_tokens], dim=1) if with_begin else out_tokens
        sound = self.decode_sound(out_tokens)
        return sound

    # logging and utils

    def log_metrics_and_samples(self, loss, batch, batch_idx, prefix=""):
        nr = self.log_nr.get(prefix, 0)
        self.log_nr[prefix] = nr + 1
        self.log(prefix + "loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if batch_idx != 0:
            return  # log samples once per epoch
        tlogger = self.logger.experiment

        # generate continuation audio
        con_samples = self.generate_from_sound(batch, prefix_token_perc=self.log_starting_context_perc, with_begin=True)
        for i, sample in enumerate(con_samples):
            tlogger.add_audio(prefix + f"sample_con_{i}", sample, nr, self.sr)

        if not self.context_on_level and prefix == "":  # raw only for train, because it does not depend on input data
            samples = self.generate(self.log_sample_size, bs=self.log_sample_bs)
            for i, sample in enumerate(samples):
                tlogger.add_audio(prefix + f"sample_raw_{i}", sample, nr, self.sr)

    def training_step(self, batch, batch_idx, name=""):
        loss = self(batch)
        self.log_metrics_and_samples(loss, batch, batch_idx, name)
        return loss

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, "test_")

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, "val_")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        # TODO scheduler on val_loss_epoch
        # sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
        # return [opt], [sched]
        return opt

