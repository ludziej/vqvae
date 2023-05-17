import torch
from functools import partial

from environment.generic_model import GenericModel
from autoencoder.modules.backbone import WavAutoEncoder
from autoencoder.modules.diff_condition import DiffusionConditioning
from autoencoder.modules.diffusion import Diffusion
from autoencoder.modules.diffusion_stats import DiffusionStats
from optimization.opt_maker import get_optimizer
import gc
from utils.misc import default


class DiffusionUnet(GenericModel):
    def __init__(self, autenc_params, diff_params, condition_params, logger_type, logger,
                 log_intervals, stats_momentum, **params):
        super().__init__(my_logger=logger, **params)
        self.diffusion = Diffusion(emb_width=self.preprocessing.emb_width, **diff_params)
        self.diffusion_cond = DiffusionConditioning(**condition_params, noise_steps=self.diffusion.noise_steps)
        self.diffusion_stats = DiffusionStats(self.diffusion.noise_steps, intervals=log_intervals,
                                              cond=self.diffusion_cond, momentum=stats_momentum,
                                              renormalize_loss=self.renormalize_loss)
        self.autenc = WavAutoEncoder(**autenc_params, base_model=self, sr=self.preprocessing.sr,
                                     cond_with_time=self.diffusion_cond.cond_with_time,
                                     condition_size=self.diffusion_cond.cond_size,
                                     input_channels=self.preprocessing.emb_width, logger_type=logger_type)
        self.set_audio_logger(lambda: self.autenc.audio_logger)

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

    def forward(self, x_in, t, time_cond=None, context_cond=None, with_metrics=False, drop_cond=None):
        if self.optimizers().optimizer.param_groups[0]["weight_decay"] > 0:
            self.optimizers().optimizer.param_groups[0]["weight_decay"] = 0
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
                                            self.prep_level + 1, bs_chunks=min(self.prep_chunks, len(latent_sound)))
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

    def evaluation_step(self, batch, batch_idx, phase="train", **params):
        x_in, context_cond, time_cond = self.preprocess(batch)
        x_noised, noise_target, t = self.noising(x_in)

        noise_pred, metrics = self(x_noised, t, with_metrics=True, context_cond=context_cond, time_cond=time_cond)
        loss, sounds_dict, metrics = self.calc_loss(noise_pred, noise_target, x_noised, x_in, t, metrics)

        assert len(metrics) == 1  # currently no support for multiple levels
        self.log_metrics_and_samples(loss=loss, metrics=metrics[0], sounds_dict=sounds_dict, t=t, batch_idx=batch_idx,
                                     phase=phase, sample_len=x_in.shape[-1], context=context_cond, time_cond=time_cond)
        return loss

    # lightning train boilerplate

    def configure_optimizers(self):
        gopt, gsched = get_optimizer(self, **self.opt_params)
        return [gopt], [gsched]

    # metrics & logging

    def heavy_logs(self, sounds_dict, t, sample_len, prefix, context=None, time_cond=None, **params):
        skip_full_sample = self.is_first_log
        super().heavy_logs()
        samples_num = min(len(t), self.log_sample_bs)
        context = context[:samples_num] if context is not None else None
        time_cond = time_cond[:samples_num] if time_cond is not None else None
        samples, sample_metrics = self.sample(samples_num, length=sample_len, context_cond=context,
                                              time_cond=time_cond, skip_full_sample=skip_full_sample)
        self.audio_logger.log_add_metrics(sample_metrics, prefix)

        sample_name = lambda ntype, ti, b, i: f"samples/{i}/{ntype}/{b}_" \
                                              f"{self.diffusion_stats.get_sample_info(b, t=ti, context=context)}"
        self.audio_logger.plot_spec_as(samples, lambda i: f"generated_specs/{i}", prefix)
        self.audio_logger.log_sounds(samples, partial(sample_name, "generated", None), prefix)

        if self.sample_cfgw is not None:
            samples, sample_metrics = self.sample(samples_num, length=sample_len, context_cond=context,
                                                  time_cond=time_cond, cfg_weight=self.sample_cfgw,
                                                  skip_full_sample=skip_full_sample)
            self.audio_logger.log_add_metrics({f"{k}_cfg": v for k, v in sample_metrics.items()}, prefix)
            self.audio_logger.plot_spec_as(samples, lambda i: f"generated_specs/cfg/{i}", prefix)
            self.audio_logger.log_sounds(samples, partial(sample_name, f"cfg={self.sample_cfgw}", None), prefix)

        for name, latent_sound in sounds_dict[0].items():
            sound = self.postprocess(latent_sound[:self.max_logged_sounds])
            self.audio_logger.plot_spec_as(sound, lambda i: f"spec_{i}/{name}", prefix)
            self.audio_logger.log_sounds(sound, partial(sample_name, name, t), prefix)

        gc.collect()