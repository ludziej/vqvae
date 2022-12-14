import torch
import torch.nn as nn
import numpy as np
import statistics
from pytorch_lightning import LightningModule
from generator.modules.performer import Performer
from vqvae.model import WavAutoEncoder
import torch.nn.functional as F
from performer_pytorch.autoregressive_wrapper import top_k, repetition_penalty_fn
from optimization.scheduler import ReduceLROnPlateauWarmup
from optimization.normalization import CustomNormalization
from utils.misc import time_run
import tqdm
from generator.modules.conditioner import Conditioner, GenerationParams
from optimization.opt_maker import get_lr_scheduler
from generator.modules.fast_transformer import FastTransformer
from typing import Dict
from functools import partial


class LevelGenerator(LightningModule):
    def __init__(self, preprocessing: WavAutoEncoder, level: int, log_sample_size: int, context_on_level: int,
                 dim: int, depth: int, heads: int, lr: float, start_gen_sample_len: int,
                 log_starting_context_perc: int, log_context_time: float, n_ctx: int, feature_redraw_interval: int,
                 pos_init_scale: int, bins_init_scale: float, dim_head: int, norm_type: bool,
                 conds_kwargs: dict, init_bins_from_vqvae: bool, share_in_out_embedding: bool,
                 conditioning_dropout: float, log_interval, token_dim: int, scheduler_type: str, pos_enc_type: str,
                 pos_enc_lvl_over_bit: int, opt_params, conditioning_concat, prep_on_cpu, use_start_token_layer,
                 use_fasttransformer, feature_map_dims, ff_mult, acc_levels, rezero, label_smoothing,
                 attn_dropout, **params):
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
        self.label_smoothing = label_smoothing
        self.pos_enc_type = pos_enc_type
        self.share_in_out_embedding = share_in_out_embedding
        self.log_interval = log_interval
        self.prep_on_cpu = prep_on_cpu
        self.opt_params = opt_params
        self.use_start_token_layer = use_start_token_layer
        self.acc_levels = acc_levels
        self.attn_dropout = attn_dropout
        self.sr = preprocessing.sr
        self.log_sample_bs, self.log_sample_size = log_sample_size
        preprocessing.freeze()
        self.preprocessing = [preprocessing] if self.prep_on_cpu else nn.ModuleList([preprocessing])
        self.bins = preprocessing.l_bins
        self.is_first_batch = True
        self.eos_token = None
        self.my_logger = preprocessing.my_logger
        self.log_nr = {"val_": 0, "": 0, "test_": 0}

        trans_args = dict(dim=dim, depth=depth, heads=heads, dim_head=dim_head, ff_mult=ff_mult,
                          feature_redraw_interval=feature_redraw_interval)
        self.transformer = FastTransformer(feature_map_dims=feature_map_dims, **trans_args) if use_fasttransformer \
            else Performer(causal=True, use_rezero=rezero, nb_features=feature_map_dims, **trans_args)

        self.sample_len = preprocessing.tokens_to_samples_num(self.n_ctx, self.level)
        assert not init_bins_from_vqvae or self.token_dim == preprocessing.emb_width
        assert self.use_start_token_layer or self.token_dim == self.dim
        self.start_layer = nn.Sequential(nn.Linear(self.token_dim, self.dim), nn.ReLU()) \
            if self.use_start_token_layer else nn.Sequential()
        z_shapes = preprocessing.samples_num_to_tokens(self.sample_len)
        z_shapes = [(z_shape[0] * self.n_ctx // z_shapes[self.level][0],) for z_shape in z_shapes]

        self.out_token_distr = np.zeros(self.bins)
        self.in_token_distr = np.zeros(self.bins)
        self.token_log_quantiles = 10
        self.training_started = set()
        self.my_logger.info(str(self))

        self.final_layer_norm = CustomNormalization(dim, norm_type=norm_type)

        self.conditioner = Conditioner(conds_kwargs=conds_kwargs, z_shapes=z_shapes, bins=self.bins,
                                       pos_enc_type=self.pos_enc_type, pos_enc_lvl_over_bit=pos_enc_lvl_over_bit,
                                       bins_init_scale=self.bins_init_scale, level=self.level, token_dim=self.token_dim,
                                       conditioning_dropout=conditioning_dropout, preprocessing=preprocessing,
                                       init_bins_from_vqvae=init_bins_from_vqvae, context_on_level=context_on_level,
                                       conditioning_concat=conditioning_concat)
        to_out = nn.Linear(dim, self.bins, bias=False)
        if self.share_in_out_embedding:
            to_out.weight = self.conditioner.x_emb.weight
        self.to_out = to_out

    def __str__(self):
        return f"Generator level {self.level} with n_ctx={self.n_ctx} and sample len={self.sample_len}"\
               f" that last {self.sample_len/self.sr:.3} s.)"

    def get_transformer_logits(self, x_emb):
        x = self.start_layer(x_emb)
        x = self.transformer(x)
        x = self.final_layer_norm(x)
        x = self.to_out(x)
        return x

    def forward(self, sound: torch.Tensor, gen_params: GenerationParams = None) -> (torch.Tensor, Dict):
        prep_time, (tokens, up_tokens) = time_run(lambda: self.get_tokens(sound))
        embedding = self.conditioner.get_conditioned_emb(tokens, up_tokens, gen_params)
        loss, metrics = self.autoregressive_forward_loss(embedding, tokens)
        metrics["prep_time"] = prep_time
        return loss, metrics

    @torch.no_grad()
    def get_accs(self, y, y_true):
        probs = torch.softmax(y, dim=-1)
        indicies = probs.sort(dim=-1, descending=True)[1]
        y_broad = y_true.reshape(-1, 1).repeat(1, y.shape[1])
        real_ranks = torch.where(indicies == y_broad)[1]
        accs = {f"token_acc/top_{l}": torch.mean((real_ranks < l).float()) for l in self.acc_levels}
        return accs

    def autoregressive_forward_loss(self, embedding, tokens) -> torch.Tensor:
        x_in = embedding[:, :-1]
        x_tok = tokens[:, 1:]
        out = self.get_transformer_logits(x_in)
        self.in_token_distr += self.calc_token_distr(x_tok)
        self.out_token_distr += self.calc_token_distr(torch.argmax(out, dim=-1))

        assert out.shape[:2] == x_in.shape[:2] and out.shape[2] == self.bins
        y, y_true = out.reshape(-1, self.bins), x_tok.reshape(-1)
        loss = F.cross_entropy(y, y_true, label_smoothing=self.label_smoothing)
        metrics = self.get_accs(y, y_true)
        return loss, metrics

    @torch.no_grad()
    def get_tokens(self, sound):
        if self.prep_on_cpu:
            if self.preprocessing[0].device.type != "cpu":
                self.my_logger.info("Moving preprocessing to CPU")
                self.preprocessing[0] = self.preprocessing[0].to("cpu")
            sound = sound.to("cpu")
        endlevel = self.level + 1 if not self.context_on_level else self.level + 2
        tokens = [x.detach() for x in self.preprocessing[0].encode(sound, start_level=self.level, end_level=endlevel)]
        tokens = [t.to(self.device) for t in tokens] if self.prep_on_cpu else tokens
        tokens, up_tokens = tokens if self.context_on_level else (tokens[0], None)
        if not self.prep_on_cpu:
            torch.cuda.empty_cache()
        return tokens, up_tokens

    # generation

    def logits_to_sample(self, logits, prev_out, temperature=1., filter_logits_fn=top_k, filter_thres=0.9,
                 repetition_penalty=1.0, repetition_penalty_ctx=32):
        if repetition_penalty > 1.0:
            logits = repetition_penalty_fn(logits, prev_out[:, -repetition_penalty_ctx:], theta=repetition_penalty)
        filtered_logits = filter_logits_fn(logits, thres=filter_thres)
        probs = F.softmax(filtered_logits / temperature, dim=-1)
        sample = torch.multinomial(probs, 1, replacement=True)
        return sample

    @torch.no_grad()
    def autoregressive_generate_prior(self, start_tokens, seq_len, conditioning=None, with_tqdm=False,
                                      **sampling_kwargs):
        was_training = self.transformer.training
        b, t = start_tokens.shape
        self.transformer.eval()
        tokens = start_tokens
        tokens_emb = self.conditioner.x_emb(start_tokens)
        x_emb_cond = None

        for i in tqdm.trange(t, seq_len, desc=f"Generating tokens [l{self.level}]") if with_tqdm else range(t, seq_len):
            ctx_start = max(i - self.n_ctx + 1, 0)  # +1 because we effectively train in (n_ctx - 1) size

            # TODO all this conditioning can be done once per element, if positional_embedding is not absolute
            x_emb_cond = self.conditioner.get_autoregressive_conditioning(x_emb_cond, tokens_emb, ctx_start, i,
                                                                          conditioning, seq_len)

            logits = self.get_transformer_logits(x_emb_cond)[:, -1, :]
            sample_token = self.logits_to_sample(logits, prev_out=tokens, **sampling_kwargs)
            sample_emb = self.conditioner.x_emb(sample_token)

            tokens = torch.cat((tokens, sample_token), dim=1)
            tokens_emb = torch.cat((tokens_emb, sample_emb), dim=1)
            if self.eos_token is not None and (sample_token == self.eos_token).all():
                break

        tokens = tokens[:, t:]
        self.transformer.train(was_training)
        return tokens

    def decode_sound(self, tokens):
        wrong_device = self.preprocessing[0].device != self.device
        tokens = tokens.to(self.preprocessing[0].device) if wrong_device else tokens
        sound = self.preprocessing[0].decode([tokens], start_level=self.level, end_level=self.level + 1).squeeze(2)
        return sound.to(self.device) if wrong_device else sound

    def recreate_beginning(self, size, bs=1):
        return torch.randint(size, (bs, self.start_gen_sample_len), device=self.device)

    def randomize_gen_params(self, ):
        return GenerationParams()  # TODO implement

    @torch.no_grad()
    def generate(self, seq_len: int, start_random_size=10, bs=1, with_tqdm=False,
                 up_tokens=None, **sampling_kwargs):
        out_tokens = self.generate_tokens(seq_len, self.randomize_gen_params(), sampling_kwargs, up_tokens, bs,
                                          start_random_size, with_tqdm, )
        sound = self.decode_sound(out_tokens)
        return sound

    @torch.no_grad()
    def generate_tokens(self, seq_len, params: GenerationParams, sampling_kwargs, up_tokens=None,
                        bs=1, start_random_size=10, with_tqdm=False):
        beginning = self.recreate_beginning(start_random_size, bs)
        conditioning = self.conditioner.get_all_conditioning(bs, seq_len, self.device, params, up_tokens=up_tokens)
        out_tokens = self.autoregressive_generate_prior(beginning, seq_len, conditioning, with_tqdm=with_tqdm,
                                                        **sampling_kwargs)
        return out_tokens

    @torch.no_grad()
    def continue_sound(self, sound: torch.Tensor, prefix_token_perc: float, params: GenerationParams, seq_len=None,
                       with_begin=True, return_speed=False, with_tqdm=False, **sampling_kwargs):
        tokens, up_tokens = self.get_tokens(sound)
        out_tokens, beginning, runtime = self.continue_tokens(tokens, params, prefix_token_perc, up_tokens,
                                                              seq_len, with_begin, with_tqdm, **sampling_kwargs)
        if not self.prep_on_cpu:
            torch.cuda.empty_cache()

        sound = self.decode_sound(out_tokens)

        generated_part = 1 - beginning.shape[-1] / out_tokens.shape[-1]
        generated_time = sound.shape[-1] / self.sr * generated_part
        speed = runtime / generated_time
        return sound, speed if return_speed else sound

    @torch.no_grad()
    def continue_tokens(self, tokens, params, prefix_token_perc, up_tokens=None, seq_len=None, with_begin=False, with_tqdm=False,
                        **sampling_kwargs):
        bs, t_len = tokens.shape[:2]
        seq_len = seq_len if seq_len is not None else t_len
        beginning = tokens[:, :int(t_len * prefix_token_perc)]
        conditioning = self.conditioner.get_all_conditioning(bs, seq_len, self.device, params, up_tokens=up_tokens)
        runtime, out_tokens = time_run(
            lambda: self.autoregressive_generate_prior(beginning, seq_len, conditioning,
                                                       with_tqdm=with_tqdm, **sampling_kwargs))
        out_tokens = torch.cat([beginning, out_tokens], dim=1) if with_begin else out_tokens
        return out_tokens, beginning, runtime

    # logging

    def calc_token_distr(self, tokens):
        distr = torch.bincount(tokens.reshape(-1), minlength=self.bins).detach()
        distr = distr.cpu() if distr.is_cuda else distr
        return distr.numpy()

    def summarize_and_reset_tok_distr(self, token_distr):
        distr = token_distr / sum(token_distr)
        token_distr[:] = np.zeros(self.bins)
        num_zeros = sum(distr == 0) / self.bins
        quantiles = [np.min(distr)] + statistics.quantiles(distr, n=self.token_log_quantiles) + [np.max(distr)]
        return quantiles, num_zeros

    def log_token_distr(self):
        in_quant, in_zeros = self.summarize_and_reset_tok_distr(self.in_token_distr)
        out_quant, out_zeros = self.summarize_and_reset_tok_distr(self.out_token_distr)

        self.log("dead_tokens_perc/in", in_zeros, logger=True, prog_bar=True, sync_dist=True)
        self.log("dead_tokens_perc/out", out_zeros, logger=True, prog_bar=True, sync_dist=True)
        for i, (in_q, out_q) in enumerate(zip(in_quant, out_quant)):
            perc = (i / self.token_log_quantiles * 100)
            self.log(f"tok_discr_quant_in/{perc:.0f}%", in_q, logger=True, sync_dist=True)
            self.log(f"tok_discr_quant_out/{perc:.0f}%", out_q, logger=True, sync_dist=True)

    def log_metrics(self, loss, metrics, prefix=""):
        self.log_nr[prefix] = self.log_nr.get(prefix, 0) + 1
        for k, v in metrics.items():
            self.log(k, v, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(prefix + "loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.is_first_batch = False if not self.trainer.sanity_checking else self.is_first_batch

    def check_log_samples(self, batch, batch_idx, gen_params: GenerationParams, prefix=""):
        if self.is_sampling_time(prefix, batch_idx):
            self.log_samples(batch, self.log_nr.get(prefix, 0), prefix, gen_params=gen_params)

    def is_sampling_time(self, prefix, batch_idx):
        """ log samples once per log_interval and once per valid """
        return not self.is_first_batch and (
             (batch_idx == 0 and not self.trainer.sanity_checking) if prefix != "" else
             (self.trainer.current_epoch * self.trainer.num_training_batches + batch_idx) % self.log_interval == 0)

    @torch.no_grad()
    def log_samples(self, batch, nr, prefix, gen_params: GenerationParams, with_tqdm=True):
        # generate continuation audio
        tlogger = self.logger.experiment
        con_samples, speed = self.continue_sound(batch, prefix_token_perc=self.log_starting_context_perc,
                                                 params=gen_params, return_speed=True, with_begin=True,
                                                 with_tqdm=with_tqdm)
        self.log("1s_gen_time", speed, sync_dist=True, logger=True, on_step=True)
        for i, sample in enumerate(con_samples):
            tlogger.add_audio(prefix + f"sample_con_{i}", sample, nr, self.sr)

        if prefix != "":  # skip these for valid
            return
        self.log_token_distr()
        if self.context_on_level:  # skip these for upsampler
            return
        if not self.prep_on_cpu:
            torch.cuda.empty_cache()

        # raw generation logging only for train on prior, because it does not depend on input data
        samples = self.generate(self.log_sample_size, bs=self.log_sample_bs, with_tqdm=with_tqdm)
        for i, sample in enumerate(samples):
            tlogger.add_audio(prefix + f"sample_raw_{i}", sample, nr, self.sr)

    # boilerplate

    def step(self, batch, batch_idx, gen_params: GenerationParams = None, phase=""):
        gen_params = gen_params if gen_params is not None else GenerationParams()
        if phase not in self.training_started:
            self.training_started.add(phase)
            self.my_logger.info(f"{(phase or 'train')} loop started - first batch arrived")
        batch, = batch
        assert batch.shape[1] == self.sample_len
        self.check_log_samples(batch, batch_idx, gen_params=gen_params, prefix=phase)

        loss, metrics = self(batch, gen_params)
        self.log_metrics(loss, metrics, prefix=phase)
        return loss

    def training_step(self, batch, batch_idx, gen_params: GenerationParams = None):
        return self.step(batch, batch_idx, phase="", gen_params=gen_params)

    def test_step(self, batch, batch_idx, gen_params: GenerationParams = None):
        return self.step(batch, batch_idx, phase="test_", gen_params=gen_params)

    def validation_step(self, batch, batch_idx, gen_params: GenerationParams = None):
        return self.step(batch, batch_idx, phase="val_", gen_params=gen_params)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=self.opt_params["adam_betas"],
                                weight_decay=self.opt_params["adam_weight_decay"])
        if self.scheduler_type == "none":
            return opt
        elif self.scheduler_type == "plateau":
            scheduler = ReduceLROnPlateauWarmup(opt, starting_lr=self.lr, warmup_time=self.opt_params["lr_warmup"],
                                                logger=self.my_logger, patience=self.opt_params["sch_patience"],
                                                factor=self.opt_params["sch_factor"])
            return ([opt], [{
                        'scheduler': scheduler,
                        'interval': 'step',
                        'frequency': 1,
                        'monitor': 'loss_step',
                        'reduce_on_plateau': True
                }])
        elif self.scheduler_type == "step":
            return ([opt], [{
                'scheduler': get_lr_scheduler(opt, **self.opt_params),
                'interval': 'step',
                'frequency': 1,
            }])
        else:
            raise Exception(f"Unknown scheduler_type = {self.scheduler_type}")
