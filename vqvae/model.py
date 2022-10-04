import numpy as np
import torch
import torch as t
import torch.nn as nn
from pytorch_lightning import LightningModule

from vqvae.encdec import Encoder, Decoder
from vqvae.bottleneck import NoBottleneck, Bottleneck
from utils.old_ml_utils.misc import average_metrics, assert_shape
from utils.old_ml_utils.audio_utils import spectral_convergence, audio_postprocess, norm
from optimization.opt_maker import get_optimizer
from vqvae.helpers import calculate_strides, _loss_fn, multispectral_loss_util, spectral_loss_util
from vqvae.adversarial.trainer import AdversarialTrainer


class VQVAEGenerator(nn.Module):
    def __init__(self, encoders: nn.ModuleList, decoders: nn.ModuleList, bottleneck: nn.Module):
        super().__init__()
        self.encoders = encoders
        self.decoders = decoders
        self.bottleneck = bottleneck


class VQVAE(LightningModule):
    def __init__(self, input_channels, levels, downs_t, strides_t, loss_fn, norm_before_vqvae, fixed_commit, logger,
                 emb_width, l_bins, mu, commit, spectral, multispectral, forward_params, multipliers, use_bottleneck,
                 adv_params, log_interval, prenorm_normalisation, prenorm_loss_weight, skip_valid_logs, **params):
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

        self.downs_t = downs_t
        self.strides_t = strides_t
        self.l_bins = l_bins
        self.commit = commit
        self.spectral = spectral
        self.multispectral = multispectral
        self.opt_params = params

        self.with_discriminator = self.adv_params["with_discriminator"]
        self.discriminator_level = self.adv_params["discriminator_level"]
        self.sr = self.forward_params["sr"] = params["sr"]
        self.downsamples = calculate_strides(strides_t, downs_t)
        self.hop_lengths = np.cumprod(self.downsamples)
        self.log_nr = {"val_": 0, "": 0, "test_": 0}
        self.my_logger.info(str(self))

        assert len(multipliers) == levels, "Invalid number of multipliers"

        def _block_kwargs(level):
            this_block_kwargs = dict(params)
            this_block_kwargs["width"] *= self.multipliers[level]
            this_block_kwargs["depth"] *= self.multipliers[level]
            return this_block_kwargs

        encoders = nn.ModuleList([Encoder(input_channels, emb_width, level + 1,
                                          downs_t[:level+1], strides_t[:level+1], **_block_kwargs(level))
                                  for level in range(levels)])
        decoders = nn.ModuleList([Decoder(input_channels, emb_width, level + 1,
                                          downs_t[:level+1], strides_t[:level+1], **_block_kwargs(level))
                                  for level in range(levels)])
        bottleneck = Bottleneck(l_bins, emb_width, mu, levels, norm_before_vqvae, fixed_commit) \
            if use_bottleneck else NoBottleneck(levels)

        self.generator: VQVAEGenerator = VQVAEGenerator(encoders, decoders, bottleneck)

        self.discriminator = AdversarialTrainer(**adv_params, **_block_kwargs(self.discriminator_level),
                                                input_channels=input_channels, level=self.discriminator_level,
                                                downs_t=downs_t[:self.discriminator_level + 1], emb_width=emb_width,
                                                strides_t=strides_t[:self.discriminator_level + 1], levels=self.levels)

    def __str__(self):
        return f"VQ-VAE with sr={self.sr} and tokens for one second: {self.get_z_lengths(1 * self.sr)}"

    def get_z_lengths(self, sample_len):
        return [(sample_len // self.hop_lengths[level],) for level in range(self.levels)]

    def samples_from_z_length(self, z_length, level):
        return z_length * self.hop_lengths[level]

    # time to encoding tokens
    def time_to_tokens(self, time: float, level: int):
        return self.get_z_lengths(time * self.sr)[level]

    def preprocess(self, x):
        # x: NTC [-1,1] -> NCT [-1,1]
        assert len(x.shape) == 3
        x = x.permute(0,2,1).float()
        return x

    def postprocess(self, x):
        # x: NTC [-1,1] <- NCT [-1,1]
        x = x.permute(0,2,1)
        return x

    def _decode(self, zs, start_level=0, end_level=None):
        # Decode
        if end_level is None:
            end_level = self.levels
        assert len(zs) == end_level - start_level
        xs_quantised = self.generator.bottleneck.decode(zs, start_level=start_level, end_level=end_level)
        assert len(xs_quantised) == end_level - start_level

        # Use only lowest level
        decoder, x_quantised = self.generator.decoders[start_level], xs_quantised[0:1]
        x_out = decoder(x_quantised, all_levels=False)
        x_out = self.postprocess(x_out)
        return x_out

    def decode(self, zs, start_level=0, end_level=None, bs_chunks=1):
        z_chunks = [t.chunk(z, bs_chunks, dim=0) for z in zs]
        x_outs = []
        for i in range(bs_chunks):
            zs_i = [z_chunk[i] for z_chunk in z_chunks]
            x_out = self._decode(zs_i, start_level=start_level, end_level=end_level)
            x_outs.append(x_out)
        return t.cat(x_outs, dim=0)

    def _encode(self, x, start_level=0, end_level=None):
        # Encode
        if end_level is None:
            end_level = self.levels
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.generator.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])
        zs = self.generator.bottleneck.encode(xs)
        return zs[start_level:end_level]

    def encode(self, x, start_level=0, end_level=None, bs_chunks=1):
        x_chunks = t.chunk(x, bs_chunks, dim=0)
        zs_list = []
        for x_i in x_chunks:
            zs_i = self._encode(x_i, start_level=start_level, end_level=end_level)
            zs_list.append(zs_i)
        zs = [t.cat(zs_level_list, dim=0) for zs_level_list in zip(*zs_list)]
        return zs

    def sample(self, n_samples, sample_len):
        zs = [t.randint(0, self.l_bins, size=(n_samples, *z_shape), device='cuda')
              for z_shape in self.get_z_lengths(sample_len)]
        return self.decode(zs)

    def forward(self, x_in):
        x_encoded = [encoder(x_in)[-1] for encoder in self.generator.encoders]
        zs, xs_quantised, commit_losses, prenorms, quantiser_metrics = self.generator.bottleneck(x_encoded)

        x_outs = [decoder(xs_quantised[level:level+1], all_levels=False)
                  for level, decoder in enumerate(self.generator.decoders)]
        [assert_shape(x_out, x_in.shape) for x_out in x_outs]

        loss, metrics = self.calc_metrics_and_loss(commit_losses, quantiser_metrics, prenorms, x_in, x_outs)
        return loss, metrics, x_outs

    def training_step(self, batch, batch_idx, optimizer_idx=-1):
        x_in = self.preprocess(batch[0])
        loss, metrics, x_outs = self(x_in)
        loss += self.discriminator.training_step(metrics, optimizer_idx, x_in, x_outs, batch_idx, self.current_epoch)
        self.log_metrics_and_samples(loss, metrics, x_in, x_outs, batch_idx)
        return loss

    def test_step(self, batch, batch_idx):
        x_in = self.preprocess(batch[0])
        loss, metrics, x_outs = self(x_in)
        if not self.skip_valid_logs:
            self.log_metrics_and_samples(loss, metrics, x_in, x_outs, batch_idx, prefix="test_")
        return loss

    def validation_step(self, batch, batch_idx):
        x_in = self.preprocess(batch[0])
        loss, metrics, x_outs = self(x_in)
        if not self.skip_valid_logs:
            self.log_metrics_and_samples(loss, metrics, x_in, x_outs, batch_idx, prefix="val_")
        return loss

    def configure_optimizers(self):
        gopt, gsched = get_optimizer(self.generator, **self.opt_params)
        if not self.with_discriminator:
            return [gopt], [gsched]
        # all parameters are given to the discriminator optimizer, because it optimizes also generator compression
        # but detaches generator output before discriminator
        dopt, dsched = get_optimizer(self, **self.opt_params)
        return [gopt, dopt], [gsched, dsched]

    # metrics

    def calc_metrics_and_loss(self, commit_losses, quantiser_metrics, prenorms, x_in, x_outs):
        metrics = {}
        hps = self.forward_params
        x_target = audio_postprocess(self.postprocess(x_in).float(), hps)

        metrics["x_norm/in"] = torch.mean(norm(x_target))
        for i, xo in enumerate(x_outs):
            metrics[f"x_norm/out_lvl_{i + 1}"] = torch.mean(norm(xo))

        x_outs = [self.postprocess(x_out) for x_out in x_outs]
        recons_loss = t.zeros(()).to(x_in.device)
        spec_loss = t.zeros(()).to(x_in.device)
        multispec_loss = t.zeros(()).to(x_in.device)
        for level in reversed(range(self.levels)):
            x_out = audio_postprocess(x_outs[level], hps)
            this_recons_loss = _loss_fn(self.loss_fn, x_target, x_out, hps)
            this_spec_loss = spectral_loss_util(self.forward_params, x_target, x_out)
            this_multispec_loss = multispectral_loss_util(self.forward_params, x_target, x_out)
            metrics[f'recons_loss/lvl_{level + 1}'] = this_recons_loss
            metrics[f'spectral_loss/lvl_{level + 1}'] = this_spec_loss
            metrics[f'multispectral_loss/lvl_{level + 1}'] = this_multispec_loss
            metrics[f'commit_loss/lvl_{level + 1}'] = commit_losses[level]
            recons_loss += this_recons_loss
            spec_loss += this_spec_loss
            multispec_loss += this_multispec_loss
        commit_loss = sum(commit_losses)
        prenorm_loss = torch.sqrt(torch.mean((torch.stack(prenorms) - self.prenorm_normalisation)**2)) \
            if self.prenorm_normalisation else 0

        loss = recons_loss + self.spectral * spec_loss + self.multispectral * multispec_loss +\
               self.commit * commit_loss + prenorm_loss * self.prenorm_loss_weight

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
                "multispectral_loss/total": multispec_loss, "commit_loss/total": commit_loss})
            for key, val in metrics.items():
                metrics[key] = val.detach()
        return loss, metrics

    def log_metrics_and_samples(self, loss, metrics, batch, batch_outs, batch_idx, prefix=""):
        self.log(prefix + "loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for name, val in metrics.items():
            self.log(prefix + name, val,  on_step=True, on_epoch=True, logger=True)
        if batch_idx % self.log_interval != 0:
            return  # log samples once per interval
        nr = self.log_nr.get(prefix, 0)
        self.log_nr[prefix] = nr + 1
        tlogger = self.logger.experiment

        for i, xin in enumerate(batch):
            tlogger.add_audio(prefix + f"sample_{i}/in", xin, nr, self.sr)

        for level, xouts in enumerate(batch_outs):
            for i, out in enumerate(xouts):
                tlogger.add_audio(prefix + f"sample_{i}/out_lvl_{level + 1}[{}]", out, nr, self.sr, self.global_rank)
