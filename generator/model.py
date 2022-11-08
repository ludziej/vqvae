import torch
import torch.nn as nn
import numpy as np
import statistics
from pytorch_lightning import LightningModule
from generator.modules.performer import Performer
from vqvae.model import VQVAE
import torch.nn.functional as F
from performer_pytorch.autoregressive_wrapper import top_k, repetition_penalty_fn
from optimization.scheduler import ReduceLROnPlateauWarmup
from optimization.normalization import CustomNormalization
from utils.misc import time_run
import tqdm
from generator.modules.conditioner import Conditioner, GenerationParams


class LevelGenerator(LightningModule):
    def __init__(self, vqvae: VQVAE, level: int, log_sample_size: int, context_on_level: int,
                 dim: int, depth: int, heads: int,  lr: float, start_gen_sample_len: int,
                 log_starting_context_perc: int, log_context_time: float, n_ctx: int, feature_redraw_interval: int,
                 pos_init_scale: int, bins_init_scale: float, dim_head: int, norm_type: bool,
                 conds_kwargs: dict, init_bins_from_vqvae: bool, layer_for_logits: bool, conditioning_dropout: float,
                 warmup_time: int, sch_patience: int, sch_factor: int, log_interval, token_dim: int,
                 scheduler_type: str, pos_enc_type: str, **params):
        super().__init__()
        self.n_ctx = n_ctx
        self.level = level
        self.start_gen_sample_len = start_gen_sample_len
        self.lr = lr
        self.dim = dim
        self.token_dim = token_dim
        self.max_seq_len = n_ctx
        self.context_on_level = context_on_level
        self.log_starting_context_perc = log_starting_context_perc
        self.log_context_time = log_context_time
        self.pos_init_scale = pos_init_scale
        self.bins_init_scale = bins_init_scale
        self.scheduler_type = scheduler_type
        self.pos_enc_type = pos_enc_type
        self.init_bins_from_vqvae = init_bins_from_vqvae
        self.layer_for_logits = layer_for_logits
        self.warmup_time = warmup_time
        self.sch_patience = sch_patience
        self.sch_factor = sch_factor
        self.log_interval = log_interval
        self.sr = vqvae.sr
        self.log_sample_bs, self.log_sample_size = log_sample_size
        self.preprocessing: VQVAE = vqvae
        self.preprocessing.freeze()
        self.bins = self.preprocessing.l_bins
        self.is_first_batch = True
        self.eos_token = None
        self.conditioning_concat = False
        self.my_logger = self.preprocessing.my_logger
        self.log_nr = {"val_": 0, "": 0, "test_": 0}

        self.transformer = Performer(causal=True, dim=dim, depth=depth, heads=heads, dim_head=dim_head,
                                     feature_redraw_interval=feature_redraw_interval)

        self.sample_len = self.preprocessing.tokens_to_samples_num(self.n_ctx, self.level)
        self.start_layer = nn.ModuleList([nn.Linear(self.token_dim, self.dim), nn.ReLU()])
        z_shapes = self.preprocessing.samples_num_to_tokens(self.sample_len)
        z_shapes = [(z_shape[0] * self.n_ctx // z_shapes[self.level][0],) for z_shape in z_shapes]

        self.token_distr = np.zeros(self.bins)
        self.token_log_quantiles = 10
        self.training_started = set()
        self.my_logger.info(str(self))

        if self.layer_for_logits:
            self.final_layer_norm = CustomNormalization(dim, norm_type=norm_type)
            self.to_out = nn.Linear(dim, self.bins)

        self.conditioner = Conditioner(conds_kwargs=conds_kwargs, z_shapes=z_shapes,
                                       conditioning_dropout=conditioning_dropout)

    def __str__(self):
        return f"Upsampler level {self.level} with n_ctx={self.n_ctx} and tokens={self.sample_len}"\
               f" that last {self.sample_len/self.sr:.3} s.)"

    def get_vqvae_bins(self, level):
        return self.preprocessing.generator.bottleneck.level_blocks[level].k.detach()

    def get_transformer_logits(self, x_emb):
        x = self.start_layer[1](self.start_layer[0](x_emb))
        x = self.transformer(x)
        if self.layer_for_logits:
            x = self.final_layer_norm(x)
            x = self.to_out(x)
            return x
        else:
            # TODO implement some version using embedding location for classification task, instead of just linear layer
            raise NotImplementedError("currently no method of token classification other then plain layer")

    def forward(self, sound: torch.Tensor, gen_params: GenerationParams = None) -> torch.Tensor:
        tokens, up_tokens = self.get_tokens(sound)
        embedding = self.get_conditioned_emb(tokens, up_tokens, gen_params)
        loss = self.autoregressive_forward_loss(embedding, tokens)
        return loss

    def autoregressive_forward_loss(self, embedding, tokens) -> torch.Tensor:
        x_in = embedding[:, :-1]
        x_tok = tokens[:, 1:]
        out = self.get_transformer_logits(x_in)
        self.append_token_distr(out)

        assert out.shape[:2] == x_in.shape[:2] and out.shape[2] == self.bins
        loss = F.cross_entropy(out.reshape(-1, self.bins), x_tok.reshape(-1))
        return loss

    def get_tokens(self, sound):
        endlevel = self.level + 1 if not self.context_on_level else self.level + 2
        tokens = [x.detach() for x in self.preprocessing.encode(sound, start_level=self.level, end_level=endlevel)]
        tokens, up_tokens = tokens if self.context_on_level else (tokens[0], None)
        return tokens, up_tokens

    # conditionals

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
    def autoregressive_generate_prior(self, start_tokens, seq_len, conditioning=None, with_tqdm=False,
                                      **sampling_kwargs):
        was_training = self.transformer.training
        b, t = start_tokens.shape
        self.transformer.eval()
        tokens = start_tokens
        tokens_emb = self.x_emb(start_tokens)

        for i in tqdm.trange(t, seq_len, desc=f"Generating tokens [{self.level}]") if with_tqdm else range(t, seq_len):
            ctx_start = max(i - self.n_ctx + 1, 0)  # +1 because we effectively train in (n_ctx - 1) size

            # TODO all this conditioning can be done once per element, if positional_embedding is not absolute
            x_emb_cond = self.join_conditioning(tokens_emb, token_interv=(ctx_start, i), **conditioning)

            logits = self.get_transformer_logits(x_emb_cond)[:, -1, :]
            sample_token = self.logits_to_sample(logits, prev_out=tokens, **sampling_kwargs)
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
    def generate(self, seq_len: int, params: GenerationParams, start_random_size=10, bs=1, with_tqdm=False,
                 up_tokens=None, **sampling_kwargs):
        out_tokens = self.generate_tokens(seq_len, params, sampling_kwargs, bs, start_random_size, with_tqdm,
                                          up_tokens=up_tokens)
        sound = self.decode_sound(out_tokens)
        return sound

    @torch.no_grad()
    def generate_tokens(self, seq_len, params: GenerationParams, sampling_kwargs, up_tokens=None,
                        bs=1, start_random_size=10, with_tqdm=False):
        beginning = self.recreate_beginning(start_random_size, bs)
        conditioning = self.get_all_conditioning(bs, seq_len, params, up_tokens=up_tokens)
        out_tokens = self.autoregressive_generate_prior(beginning, seq_len, conditioning, with_tqdm=with_tqdm,
                                                        **sampling_kwargs)
        return out_tokens

    @torch.no_grad()
    def continue_sound(self, sound: torch.Tensor, prefix_token_perc: float, params: GenerationParams, seq_len=None,
                       with_begin=True, return_speed=False, with_tqdm=False, **sampling_kwargs):
        tokens, up_tokens = self.get_tokens(sound)
        beginning, out_tokens, runtime, sound = self.continue_tokens(tokens, up_tokens, params, prefix_token_perc,
                                                                     seq_len, with_begin, with_tqdm, **sampling_kwargs)

        generated_part = 1 - beginning.shape[-1] / out_tokens.shape[-1]
        generated_time = sound.shape[-1] / self.preprocessing.sr * generated_part
        speed = runtime / generated_time
        return sound, speed if return_speed else sound

    @torch.no_grad()
    def continue_tokens(self, tokens, params, prefix_token_perc, up_tokens=None, seq_len=None, with_begin=False, with_tqdm=False,
                        **sampling_kwargs):
        bs, t_len = tokens.shape[:2]
        seq_len = seq_len if seq_len is not None else t_len
        beginning = tokens[:, :int(t_len * prefix_token_perc)]
        conditioning = self.get_all_conditioning(bs, seq_len, params, up_tokens=up_tokens, )
        runtime, out_tokens = time_run(
            lambda: self.autoregressive_generate_prior(beginning, seq_len, conditioning,
                                                       with_tqdm=with_tqdm, **sampling_kwargs))
        out_tokens = torch.cat([beginning, out_tokens], dim=1) if with_begin else out_tokens
        sound = self.decode_sound(out_tokens)
        return beginning, out_tokens, runtime, sound

    # logging

    def append_token_distr(self, logits):
        tokens = torch.argmax(logits, dim=-1).reshape(-1)
        distr = torch.bincount(tokens, minlength=self.bins).detach()
        distr = distr.cpu() if distr.is_cuda else distr
        self.token_distr += distr.numpy()

    def log_token_distr_and_reset(self):
        distr = self.token_distr / sum(self.token_distr)
        self.token_distr = np.zeros(self.bins)
        num_zeros = sum(distr == 0) / self.bins
        quantiles = [np.min(distr)] + statistics.quantiles(distr, n=self.token_log_quantiles) + [np.max(distr)]

        self.log("dead_tokens_perc", num_zeros, logger=True, prog_bar=True, sync_dist=True)
        q_dict = {f"{(i / self.token_log_quantiles * 100):.0f}%": q for i, q in enumerate(quantiles)}
        self.logger.experiment.add_scalars('tok_discr_quant', q_dict)

    def is_sampling_time(self, prefix, batch_idx):
        """ log samples once per log_interval and once per valid """
        return not self.is_first_batch and (batch_idx == 0 and not self.trainer.sanity_checking if prefix != "" else
            (self.trainer.current_epoch * self.trainer.num_training_batches + batch_idx) % self.log_interval == 0)

    def log_metrics_and_samples(self, loss, batch, batch_idx, prefix=""):
        nr = self.log_nr.get(prefix, 0)
        self.log_nr[prefix] = nr + 1
        for i, pg in enumerate(self.optimizers().param_groups):
            self.log(f"lr_{i}", pg["lr"], logger=True, prog_bar=True, sync_dist=True)
        self.log(prefix + "loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if self.is_sampling_time(prefix, batch_idx):
            self.log_samples(batch, nr, prefix)
        self.is_first_batch = False if not self.trainer.sanity_checking else self.is_first_batch

    def log_samples(self, batch, nr, prefix):
        # generate continuation audio
        tlogger = self.logger.experiment
        con_samples, speed = self.continue_sound(batch, prefix_token_perc=self.log_starting_context_perc,
                                                 return_speed=True, with_begin=True)
        self.log("1s_gen_time", speed, sync_dist=True, logger=True, on_step=True)
        for i, sample in enumerate(con_samples):
            tlogger.add_audio(prefix + f"sample_con_{i}", sample, nr, self.sr)

        if prefix != "":  # skip these for valid
            return
        self.log_token_distr_and_reset()
        if self.context_on_level:  # skip these for upsampler
            return

        # raw generation logging only for train on prior, because it does not depend on input data
        samples = self.generate(self.log_sample_size, bs=self.log_sample_bs)
        for i, sample in enumerate(samples):
            tlogger.add_audio(prefix + f"sample_raw_{i}", sample, nr, self.sr)

    # boilerplate

    def step(self, batch, batch_idx, gen_params: GenerationParams = None, phase=""):
        gen_params = gen_params if gen_params is not None else GenerationParams(None, None)
        if phase not in self.training_started:
            self.training_started.add(phase)
            self.my_logger.info(f"{(phase or 'train')} loop started - first batch arrived")
        batch, = batch
        assert batch.shape[1] == self.sample_len
        loss = self(batch, gen_params)
        self.log_metrics_and_samples(loss, batch, batch_idx, phase)
        return loss

    def training_step(self, batch, batch_idx, gen_params: GenerationParams = None):
        return self.step(batch, batch_idx, phase="")

    def test_step(self, batch, batch_idx, gen_params: GenerationParams = None):
        return self.step(batch, batch_idx, phase="test_")

    def validation_step(self, batch, batch_idx, gen_params: GenerationParams = None):
        return self.step(batch, batch_idx, phase="val_")

    def configure_optimizers(self):
        if self.no_scheduler:
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        opt = torch.optim.Adam(self.parameters(), lr=self.lr / self.warmup_time)
        scheduler = ReduceLROnPlateauWarmup(opt, starting_lr=self.lr, warmup_time=self.warmup_time,
                                            logger=self.my_logger, patience=self.sch_patience, factor=self.sch_factor)
        return (
            [opt],
            [
                {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                    'monitor': 'loss_step',
                    'reduce_on_plateau': True
                }
            ]
        )
