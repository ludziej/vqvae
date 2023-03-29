import torch
from pytorch_lightning import LightningModule
from functools import partial

from vqvae.modules.backbone import WavAutoEncoder
from vqvae.model import WavCompressor
from vqvae.modules.diff_condition import DiffusionConditioning
from vqvae.modules.diffusion import Diffusion
from vqvae.modules.diffusion_stats import DiffusionStats
from optimization.opt_maker import get_optimizer
import gc
from utils.misc import default


class DiffusionUnet(LightningModule):
    def __init__(self, autenc_params, preprocessing: WavCompressor, diff_params, log_sample_bs, prep_chunks,
                 prep_level, opt_params, logger, log_interval, max_logged_sounds, no_stochastic_prep,
                 rmse_loss_weight, eps_loss_weight, condition_params, data_time_cond, data_context_cond, logger_type,
                 log_intervals, stats_momentum, renormalize_loss, sample_cfgw, **params):
        super().__init__()
        self.renormalize_loss = renormalize_loss
        self.preprocessing = preprocessing
        self.data_time_cond = data_time_cond
        self.data_context_cond = data_context_cond
        self.rmse_loss_weight = rmse_loss_weight
        self.eps_loss_weight = eps_loss_weight
        self.log_sample_bs = log_sample_bs
        self.prep_chunks = prep_chunks
        self.sample_cfgw = sample_cfgw
        self.max_logged_sounds = max_logged_sounds
        self.opt_params = opt_params
        self.prep_level = prep_level
        self.my_logger = logger
        self.log_interval = log_interval
        self.no_stochastic_prep = no_stochastic_prep
        self.skip_valid_logs = True
        self.diffusion = Diffusion(emb_width=self.preprocessing.emb_width, **diff_params)
        self.diffusion_cond = DiffusionConditioning(**condition_params, noise_steps=self.diffusion.noise_steps)
        self.diffusion_stats = DiffusionStats(self.diffusion.noise_steps, intervals=log_intervals,
                                              cond=self.diffusion_cond, momentum=stats_momentum,
                                              renormalize_loss=renormalize_loss)
        self.autenc = WavAutoEncoder(**autenc_params, base_model=self, sr=self.preprocessing.sr,
                                     cond_with_time=self.diffusion_cond.cond_with_time,
                                     condition_size=self.diffusion_cond.cond_size,
                                     input_channels=self.preprocessing.emb_width, logger_type=logger_type)

        for param in self.preprocessing.parameters():
            param.requires_grad = False

    def __str__(self):
        return f"Diffusion model on {self.preprocessing}"

    @torch.no_grad()
    def eval_no_grad(self, x_in, t, cfg_weight=None, **args):
        if not self.diffusion_cond.cls_free_guidance:
            return self(x_in, t, **args, with_metrics=False)
        cfg_weight = default(cfg_weight, self.diffusion_cond.cfg_guid_weight)
        guided = self(x_in, t, **args, with_metrics=False, drop_cond=False)
        if cfg_weight == 1:
            return guided
        unguided = self(x_in, t, **args, with_metrics=False, drop_cond=True)
        return [cfg_weight * guided[0] + (1 - cfg_weight) * unguided[0]]

    def on_after_backward(self) -> None:
        self.autenc.audio_logger.log_grads()

    def forward(self, x_in, t, time_cond=None, context_cond=None, with_metrics=False, drop_cond=None):
        conditioning = self.diffusion_cond.get_conditioning(t, time_cond=time_cond, context_cond=context_cond,
                                                            length=x_in.shape[2], drop_cond=drop_cond)
        x_predicted, _, _, metrics = self.autenc.forward(x_in, cond=conditioning)
        if with_metrics:
            return x_predicted, metrics
        return x_predicted

    @torch.no_grad()
    def preprocess(self, batch):
        sound, data_cond, time_cond = (batch[0], batch[1].get("context", None), batch[1].get("times", None)) \
            if self.data_time_cond or self.data_context_cond else (batch[0], None, None)
        b_params = dict(var_temp=0) if self.no_stochastic_prep else None
        latent = self.preprocessing.encode(sound, self.prep_level, self.prep_level + 1, bs_chunks=self.prep_chunks,
                                           b_params=b_params)[0]
        return latent, data_cond, time_cond

    @torch.no_grad()
    def postprocess(self, latent_sound):
        decoded = self.preprocessing.decode([latent_sound], self.prep_level,
                                            self.prep_level + 1, bs_chunks=self.prep_chunks)
        return decoded.permute(0, 2, 1)

    def noising(self, x):
        t = self.diffusion.sample_timesteps(x.shape[0])
        x_noised, noise_target = self.diffusion.noise_images(x, t)
        return x_noised, noise_target, t

    @torch.no_grad()
    def sample(self, bs, length, **kwargs):
        latent_samples = self.diffusion.sample(self, bs, length=length, **kwargs)
        sample_norm = self.diffusion_stats.bnorm(latent_samples)
        samples = self.postprocess(latent_samples)
        return samples, dict(sample_norm=sample_norm)

    def calc_loss(self, preds, target, x_noised, x_in, t, metrics_l):
        loss, sounds_dict = 0, []
        assert len(preds) == 1  # wont work for multiple levels
        for metrics, pred in zip(metrics_l, preds):
            x_pred_denoised = self.diffusion.get_x0(x_noised, pred, t)
            e_loss, e_metrics = self.diffusion_stats.residual_metrics(pred, target, "e", t)
            x_loss, x_metrics = self.diffusion_stats.residual_metrics(x_pred_denoised, x_in, "x", t)
            loss += self.eps_loss_weight * e_loss + self.rmse_loss_weight * x_loss
            self.diffusion_stats.aggregate(t, metrics={**e_metrics, **x_metrics})
            metrics.update(self.diffusion_stats.get_aggr())
            sounds_dict.append(dict(x_in=x_in, x_pred_denoised=x_pred_denoised, x_noised=x_noised))
        return loss, sounds_dict, metrics_l

    def evaluation_step(self, batch, batch_idx, phase="train"):
        x_in, context_cond, time_cond = self.preprocess(batch)
        x_noised, noise_target, t = self.noising(x_in)

        noise_pred, metrics = self(x_noised, t, with_metrics=True, context_cond=context_cond, time_cond=time_cond)
        loss, sounds_dict, metrics = self.calc_loss(noise_pred, noise_target, x_noised, x_in, t, metrics)

        self.log_metrics_and_samples(loss, metrics, sounds_dict, t, batch_idx, phase, sample_len=x_in.shape[-1],
                                     context=context_cond, time_cond=time_cond)
        return loss

    # lightning train boilerplate

    def training_step(self, batch, batch_idx):
        return self.evaluation_step(batch, batch_idx, "train")

    def test_step(self, batch, batch_idx):
        return self.evaluation_step(batch, batch_idx, "test")

    def validation_step(self, batch, batch_idx):
        return self.evaluation_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        gopt, gsched = get_optimizer(self, **self.opt_params)
        return [gopt], [gsched]

    # metrics & logging

    @torch.no_grad()
    def log_metrics_and_samples(self, loss, metrics, sounds_dict, t, batch_idx, phase, sample_len, context=None,
                                time_cond=None):
        prefix = phase + "_" if phase != "train" else ""
        assert len(metrics) == 1  # currently no support for multiple levels
        self.autenc.audio_logger.log_metrics({**metrics[0], 'loss': loss}, prefix)

        if not (batch_idx % self.log_interval == 0 and self.local_rank == 0 and
                (phase == "train" or not self.skip_valid_logs)):
            return  # log samples once per interval

        samples_num = min(len(t), self.log_sample_bs)
        context = context[:samples_num] if context is not None else None
        time_cond = time_cond[:samples_num] if time_cond is not None else None
        samples, sample_metrics = self.sample(samples_num, length=sample_len, context_cond=context,
                                              time_cond=time_cond)
        self.autenc.audio_logger.log_add_metrics(sample_metrics, prefix)

        sample_name = lambda ntype, ti, b, i: f"samples/{i}/{ntype}/{b}_" \
                                              f"{self.diffusion_stats.get_sample_info(b, t=ti, context=context)}"
        self.autenc.audio_logger.plot_spec_as(samples, lambda i: f"generated_specs/{i}", prefix)
        self.autenc.audio_logger.log_sounds(samples, partial(sample_name, "generated", None), prefix)

        if self.sample_cfgw is not None:
            samples, sample_metrics = self.sample(samples_num, length=sample_len, context_cond=context,
                                                  time_cond=time_cond, cfg_weight=self.sample_cfgw)
            self.autenc.audio_logger.log_add_metrics({f"{k}_cfg": v for k, v in sample_metrics.items()}, prefix)
            self.autenc.audio_logger.plot_spec_as(samples, lambda i: f"generated_specs/cfg/{i}", prefix)
            self.autenc.audio_logger.log_sounds(samples, partial(sample_name, f"cfg={self.sample_cfgw}", None), prefix)

        for name, latent_sound in sounds_dict[0].items():
            sound = self.postprocess(latent_sound[:self.max_logged_sounds])
            self.autenc.audio_logger.plot_spec_as(sound, lambda i: f"spec_{i}/{name}", prefix)
            self.autenc.audio_logger.log_sounds(sound, partial(sample_name, name, t), prefix)

        gc.collect()
