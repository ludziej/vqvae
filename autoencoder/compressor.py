import torch
import torch as t
from pytorch_lightning import LightningModule

from autoencoder.modules.backbone import WavAutoEncoder
from utils.old_ml_utils.misc import average_metrics
from utils.old_ml_utils.audio_utils import spectral_convergence, audio_postprocess, norm
from optimization.opt_maker import get_optimizer
from autoencoder.modules.helpers import calculate_strides, _loss_fn, multispectral_loss_util, spectral_loss_util
from autoencoder.adversarial.trainer import AdversarialTrainer
import numpy as np


class WavCompressor(LightningModule):
    def __init__(self, input_channels, levels, downs_t, strides_t, loss_fn, norm_before_vqvae, fixed_commit, logger,
                 emb_width, l_bins, mu, bottleneck_lw, spectral, multispectral, forward_params, multipliers,
                 bottleneck_type, adv_params, log_interval, prenorm_normalisation, prenorm_loss_weight, skip_valid_logs,
                 rms_normalize_level, logger_type, log_vae_no_stochastic=False, use_log_grads=0, log_weights_norm=0,
                 **params):
        super().__init__()

        self.levels = levels
        self.forward_params = forward_params
        self.loss_fn = loss_fn
        self.adv_params = adv_params
        self.my_logger = logger
        self.log_interval = log_interval
        self.multipliers = multipliers
        self.prenorm_normalisation = prenorm_normalisation
        self.prenorm_loss_weight = prenorm_loss_weight
        self.skip_valid_logs = skip_valid_logs
        self.rms_normalize_level = rms_normalize_level
        self.emb_width = emb_width
        self.bottleneck_type = bottleneck_type
        self.log_vae_no_stochastic = log_vae_no_stochastic

        self.downs_t = downs_t
        self.strides_t = strides_t
        self.l_bins = l_bins
        self.bottleneck_lw = bottleneck_lw
        self.spectral = spectral
        self.multispectral = multispectral
        self.opt_params = params

        self.with_discriminator = self.adv_params["with_discriminator"]
        self.discriminator_level = self.adv_params["discriminator_level"]
        self.sr = self.forward_params["sr"] = params["sr"]
        self.downsamples = calculate_strides(strides_t, downs_t)
        self.hop_lengths = np.cumprod(self.downsamples)

        self.generator = WavAutoEncoder(self.sr, downs_t, emb_width, input_channels,
                                        l_bins, levels, mu, norm_before_vqvae, strides_t, bottleneck_type,
                                        base_model=self, multipliers=multipliers, block_params=params,
                                        skip_connections=False, fixed_commit=fixed_commit, logger_type=logger_type,
                                        use_log_grads=use_log_grads, log_weights_norm=log_weights_norm)
        adv_block = self.generator.block_kwargs(self.discriminator_level, multiply=adv_params["multiply_level"])
        self.discriminator = AdversarialTrainer(**adv_params, **adv_block,
                                                input_channels=input_channels, level=self.discriminator_level,
                                                downs_t=downs_t[:self.discriminator_level + 1], emb_width=emb_width,
                                                strides_t=strides_t[:self.discriminator_level + 1], levels=self.levels)
        self.my_logger.info(str(self))

    # helpers & api

    def __str__(self):
        return f"{self.generator.name} with sr={self.sr} " \
               f"and tokens for one second: {self.samples_num_to_tokens(1 * self.sr)}"

    def samples_num_to_tokens(self, sample_len):
        return [(sample_len // self.hop_lengths[level],) for level in range(self.levels)]

    def tokens_to_samples_num(self, z_length, level):
        return z_length * self.hop_lengths[level]

    # time to encoding tokens
    def time_to_tokens(self, time: float, level: int):
        return self.samples_num_to_tokens(time * self.sr)[level]

    def sample(self, bs, sample_len):
        zs = [t.randint(0, self.l_bins, size=(bs, *z_shape), device='cuda')
              for z_shape in self.samples_num_to_tokens(sample_len)]
        return self.decode(zs)

    # encoding & decoding

    def decode(self, zs, start_level=0, end_level=None, bs_chunks=1, **params):
        return self.generator.decode(zs, start_level, end_level, bs_chunks, **params)

    def encode(self, x, start_level=0, end_level=None, bs_chunks=1, **params):
        return self.generator.encode(x, start_level, end_level, bs_chunks, **params)

    # training & forward

    def forward(self, x_in, b_params=None):
        x_outs, bottleneck_losses, prenorms, metrics = self.generator(x_in, b_params=b_params)
        loss, metrics = self.calc_metrics_and_loss(bottleneck_losses, metrics, prenorms, x_in, x_outs)
        return loss, metrics, x_outs

    @torch.no_grad()
    def no_grad_forward(self, x_in, b_params=None):
        return self(x_in, b_params=b_params)

    def evaluation_step(self, batch, batch_idx, optimizer_idx, phase="train"):
        optimize_generator = optimizer_idx != 1
        if not optimize_generator and not self.discriminator.is_used(batch_idx, self.current_epoch, optimize_generator):
            return None
        x_in = self.generator.preprocess(batch[0])
        loss, metrics, x_outs = self(x_in) if optimize_generator else self.no_grad_forward(x_in)
        loss += self.discriminator.training_step(metrics, optimize_generator, x_in, x_outs, batch_idx,
                                                 self.current_epoch)
        self.log_metrics_and_samples(loss, metrics, x_in, x_outs, batch_idx, optimize_generator, phase)
        return loss

    # lightning train boilerplate

    def training_step(self, batch, batch_idx, optimizer_idx=-1):
        return self.evaluation_step(batch, batch_idx, optimizer_idx, "train")

    def test_step(self, batch, batch_idx, optimizer_idx=-1):
        return self.evaluation_step(batch, batch_idx, optimizer_idx, "test")

    def validation_step(self, batch, batch_idx, optimizer_idx=-1):
        return self.evaluation_step(batch, batch_idx, optimizer_idx, "val")

    def on_train_batch_end(self, outputs, batch, batch_idx: int, unused=0) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        pass

    def configure_optimizers(self):
        gopt, gsched = get_optimizer(self.generator, **self.opt_params)
        if not self.with_discriminator:
            return [gopt], [gsched]
        dopt, dsched = get_optimizer(self.discriminator, **self.opt_params)
        return [gopt, dopt], [gsched, dsched]

    # metrics & logging

    def calc_metrics_and_loss(self, bottleneck_losses, quantiser_metrics, prenorms, x_in, x_outs):
        metrics = {}
        hps = self.forward_params
        x_target = audio_postprocess(self.generator.postprocess(x_in).float(), hps)

        metrics["x_norm/in"] = torch.mean(norm(x_target))
        for i, xo in enumerate(x_outs):
            metrics[f"x_norm/out_lvl_{i + 1}"] = torch.mean(norm(xo))

        x_outs = [self.generator.postprocess(x_out) for x_out in x_outs]
        recons_loss, spec_loss, multispec_loss = [t.zeros(()).to(x_in.device) for _ in range(3)]
        for level in reversed(range(self.levels)):
            x_out = audio_postprocess(x_outs[level], hps)
            this_recons_loss = _loss_fn(self.loss_fn, x_target, x_out, hps)
            this_spec_loss = spectral_loss_util(self.forward_params, x_target, x_out)
            this_multispec_loss = multispectral_loss_util(self.forward_params, x_target, x_out)
            metrics[f'recons_loss/lvl_{level + 1}'] = this_recons_loss
            metrics[f'spectral_loss/lvl_{level + 1}'] = this_spec_loss
            metrics[f'multispectral_loss/lvl_{level + 1}'] = this_multispec_loss
            metrics[f'bottleneck_loss/lvl_{level + 1}'] = bottleneck_losses[level]
            recons_loss += this_recons_loss
            spec_loss += this_spec_loss
            multispec_loss += this_multispec_loss
        bottleneck_loss = sum(bottleneck_losses)
        prenorm_loss = torch.sqrt(torch.mean((torch.stack(prenorms) - self.prenorm_normalisation)**2)) \
            if self.prenorm_normalisation else 0

        loss = recons_loss + self.spectral * spec_loss + self.multispectral * multispec_loss + \
               self.bottleneck_lw * bottleneck_loss + prenorm_loss * self.prenorm_loss_weight

        with t.no_grad():
            for level, x_out in enumerate(x_outs):
                metrics[f"spectral_convergence/lvl_{level + 1}"] = t.mean(spectral_convergence(x_target, x_out, hps))
                metrics[f"l2_loss/lvl_{level + 1}"] = _loss_fn("l2", x_target, x_out, hps)
                metrics[f"l1_loss/lvl_{level + 1}"] = _loss_fn("l1", x_target, x_out, hps)
                metrics[f"linf_loss/lvl_{level + 1}"] = _loss_fn("linf", x_target, x_out, hps)
            for level in range(len(quantiser_metrics)):
                for key, value in quantiser_metrics[level].items():
                    metrics[f"{key}/lvl_{level}"] = value
            if self.prenorm_normalisation:
                metrics["prenorm_loss"] = prenorm_loss
            metrics.update(average_metrics(quantiser_metrics, suffix="/total"))
            metrics.update({  # add level-aggregated stats
                "recons_loss/total": recons_loss, "spectral_loss/total": spec_loss,
                "multispectral_loss/total": multispec_loss, "bottleneck_loss/total": bottleneck_loss})
            for key, val in metrics.items():
                metrics[key] = val.detach()
        return loss, metrics

    @torch.no_grad()
    def log_metrics_and_samples(self, loss, metrics, batch, batch_outs, batch_idx, optimize_generator, phase):
        prefix = phase + "_" if phase != "train" else ""
        self.generator.audio_logger.log_metrics({**metrics, 'loss': loss}, prefix)
        if not (batch_idx % self.log_interval == 0 and optimize_generator and self.local_rank == 0 and
                (phase == "train" or not self.skip_valid_logs)):
            return  # log samples once per interval

        self.generator.audio_logger.plot_spectrorams(batch, batch_outs, prefix)
        self.generator.audio_logger.log_sounds(batch, lambda b, i: f"sample_{b}/in", prefix)
        for level, xouts in enumerate(batch_outs):
            self.generator.audio_logger.log_sounds(xouts, lambda b, i: f"sample_{b}/out_lvl_{level + 1}", prefix)

        if self.log_vae_no_stochastic:
            _, _, outs_no_s = self.no_grad_forward(batch, b_params=dict(var_temp=0))
            for level, xouts in enumerate(outs_no_s):
                self.generator.audio_logger.log_sounds(xouts, lambda b, i: f"sample_{b}/out_no_s_lvl_{level + 1}", prefix)
