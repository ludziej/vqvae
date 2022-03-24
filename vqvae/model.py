import numpy as np
import torch as t
import torch.nn as nn
from pytorch_lightning import LightningModule
import logging

from vqvae.encdec import Encoder, Decoder, assert_shape
from vqvae.bottleneck import NoBottleneck, Bottleneck
from old_ml_utils.misc import average_metrics
from old_ml_utils.audio_utils import spectral_convergence, spectral_loss, multispectral_loss, audio_postprocess
from optimization.opt_maker import get_optimizer
from vqvae.helpers import calculate_strides, _loss_fn


class VQVAE(LightningModule):
    def __init__(self, input_channels, levels, downs_t, strides_t, loss_fn, norm_before_vqvae,
                 emb_width, l_bins, mu, commit, spectral, multispectral, forward_params, bottleneck_momentum,
                 multipliers, use_bottleneck=True, **params):
        super().__init__()

        self.downsamples = calculate_strides(strides_t, downs_t)
        self.hop_lengths = np.cumprod(self.downsamples)
        self.levels = levels
        self.norm_before_vqvae = norm_before_vqvae
        self.forward_params = forward_params
        self.loss_fn = loss_fn
        self.sr = params["sr"]
        assert self.sr == self.forward_params["sr"]

        assert len(multipliers) == levels, "Invalid number of multipliers"
        self.multipliers = multipliers
        def _block_kwargs(level):
            this_block_kwargs = dict(params)
            this_block_kwargs["width"] *= self.multipliers[level]
            this_block_kwargs["depth"] *= self.multipliers[level]
            return this_block_kwargs

        self.encoders = nn.ModuleList([Encoder(input_channels, emb_width, level + 1,
                                               downs_t[:level+1], strides_t[:level+1], **_block_kwargs(level))
                                       for level in range(levels)])
        self.decoders = nn.ModuleList([Decoder(input_channels, emb_width, level + 1,
                                               downs_t[:level+1], strides_t[:level+1], **_block_kwargs(level))
                                       for level in range(levels)])

        if use_bottleneck:
            self.bottleneck = Bottleneck(l_bins, emb_width, mu, levels, norm_before_vqvae, bottleneck_momentum)
        else:
            self.bottleneck = NoBottleneck(levels)

        self.downs_t = downs_t
        self.strides_t = strides_t
        self.l_bins = l_bins
        self.commit = commit
        self.spectral = spectral
        self.multispectral = multispectral
        self.opt_params = params
        self.log_nr = {"val_": 0, "": 0, "test_": 0}
        logging.info(str(self))

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
        xs_quantised = self.bottleneck.decode(zs, start_level=start_level, end_level=end_level)
        assert len(xs_quantised) == end_level - start_level

        # Use only lowest level
        decoder, x_quantised = self.decoders[start_level], xs_quantised[0:1]
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
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])
        zs = self.bottleneck.encode(xs)
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

    def forward(self, x):
        hps = self.forward_params
        loss_fn = self.loss_fn
        metrics = {}

        N = x.shape[0]

        # Encode/Decode
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])

        zs, xs_quantised, commit_losses, quantiser_metrics = self.bottleneck(xs)
        x_outs = []
        for level in range(self.levels):
            decoder = self.decoders[level]
            x_out = decoder(xs_quantised[level:level+1], all_levels=False)
            assert_shape(x_out, x_in.shape)
            x_outs.append(x_out)

        # Loss
        def _spectral_loss(x_target, x_out, hps):
            if hps.use_nonrelative_specloss:
                sl = spectral_loss(x_target, x_out, hps) / hps.bandwidth['spec']
            else:
                sl = spectral_convergence(x_target, x_out, hps)
            sl = t.mean(sl)
            return sl

        def _multispectral_loss(x_target, x_out, hps):
            sl = multispectral_loss(x_target, x_out, hps) / hps.bandwidth['spec']
            sl = t.mean(sl)
            return sl

        recons_loss = t.zeros(()).to(x.device)
        spec_loss = t.zeros(()).to(x.device)
        multispec_loss = t.zeros(()).to(x.device)
        x_target = audio_postprocess(x.float(), hps)

        for level in reversed(range(self.levels)):
            x_out = self.postprocess(x_outs[level])
            x_out = audio_postprocess(x_out, hps)
            this_recons_loss = _loss_fn(loss_fn, x_target, x_out, hps)
            this_spec_loss = _spectral_loss(x_target, x_out, hps)
            this_multispec_loss = _multispectral_loss(x_target, x_out, hps)
            metrics[f'recons_loss_l{level + 1}'] = this_recons_loss
            metrics[f'spectral_loss_l{level + 1}'] = this_spec_loss
            metrics[f'multispectral_loss_l{level + 1}'] = this_multispec_loss
            recons_loss += this_recons_loss
            spec_loss += this_spec_loss
            multispec_loss += this_multispec_loss

        commit_loss = sum(commit_losses)
        loss = recons_loss + self.spectral * spec_loss + self.multispectral * multispec_loss + self.commit * commit_loss

        with t.no_grad():
            sc = t.mean(spectral_convergence(x_target, x_out, hps))
            l2_loss = _loss_fn("l2", x_target, x_out, hps)
            l1_loss = _loss_fn("l1", x_target, x_out, hps)
            linf_loss = _loss_fn("linf", x_target, x_out, hps)

        quantiser_metrics = average_metrics(quantiser_metrics)

        metrics.update(dict(
            recons_loss=recons_loss,
            spectral_loss=spec_loss,
            multispectral_loss=multispec_loss,
            spectral_convergence=sc,
            l2_loss=l2_loss,
            l1_loss=l1_loss,
            linf_loss=linf_loss,
            commit_loss=commit_loss,
            **quantiser_metrics))

        for key, val in metrics.items():
            metrics[key] = val.detach()

        return x_out, loss, metrics, x_outs

    def log_metrics_and_samples(self, loss, metrics, batch, batch_outs, batch_idx, prefix=""):
        self.log(prefix + "loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for name, val in metrics.items():
            self.log(prefix + name, val,  on_step=True, on_epoch=True, logger=True)
        if batch_idx != 0:
            return  # log samples once per epoch
        nr = self.log_nr.get(prefix, 0)
        self.log_nr[prefix] = nr + 1
        tlogger = self.logger.experiment

        for i, xin in enumerate(batch):
            tlogger.add_audio(prefix + f"sample_in_{i}", xin, nr, self.sr)

        for level, xouts in enumerate(batch_outs):
            for i, out in enumerate(xouts):
                tlogger.add_audio(prefix + f"sample_out_{i}_lvl_{level}", out, nr, self.sr)

    def training_step(self, batch, batch_idx):
        x_out, loss, metrics, x_outs = self(batch)
        self.log_metrics_and_samples(loss, metrics, batch, x_outs, batch_idx)
        return loss

    def test_step(self, batch, batch_idx):
        x_out, loss, metrics, x_outs = self(batch)
        self.log_metrics_and_samples(loss, metrics, batch, x_outs, batch_idx, prefix="test_")
        return loss

    def validation_step(self, batch, batch_idx):
        x_out, loss, metrics, x_outs = self(batch)
        self.log_metrics_and_samples(loss, metrics, batch, x_outs, batch_idx, prefix="val_")
        return loss

    def configure_optimizers(self):
        opt, sched, scalar = get_optimizer(self, **self.opt_params)
        self.scalar = scalar
        return [opt], [sched]
