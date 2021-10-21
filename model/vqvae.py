import numpy as np
import torch
import torch as t
import torch.nn as nn
from pytorch_lightning import LightningModule
import torchaudio
from torchaudio.sox_effects import  apply_effects_tensor

from model.encdec import Encoder, Decoder, assert_shape
from model.bottleneck import NoBottleneck, Bottleneck
from ml_utils.logger import average_metrics
from ml_utils.audio_utils import spectral_convergence, spectral_loss, multispectral_loss, audio_postprocess
from optimization.opt_maker import get_optimizer

repr = lambda x: "{}, {}, {}".format(x.shape, x.type(), x.device)

def dont_update(params):
    for param in params:
        param.requires_grad = False

def update(params):
    for param in params:
        param.requires_grad = True

def calculate_strides(strides, downs):
    return [stride ** down for stride, down in zip(strides, downs)]

def _loss_fn(loss_fn, x_target, x_pred, hps):
    if loss_fn == 'l1':
        return t.mean(t.abs(x_pred - x_target)) / hps.bandwidth['l1']
    elif loss_fn == 'l2':
        return t.mean((x_pred - x_target) ** 2) / hps.bandwidth['l2']
    elif loss_fn == 'linf':
        residual = ((x_pred - x_target) ** 2).reshape(x_target.shape[0], -1)
        values, _ = t.topk(residual, hps.linf_k, dim=1)
        return t.mean(values) / hps.bandwidth['l2']
    elif loss_fn == 'lmix':
        loss = 0.0
        if hps.lmix_l1:
            loss += hps.lmix_l1 * _loss_fn('l1', x_target, x_pred, hps)
        if hps.lmix_l2:
            loss += hps.lmix_l2 * _loss_fn('l2', x_target, x_pred, hps)
        if hps.lmix_linf:
            loss += hps.lmix_linf * _loss_fn('linf', x_target, x_pred, hps)
        return loss
    else:
        assert False, f"Unknown loss_fn {loss_fn}"


class VQVAE(LightningModule):
    def __init__(self, input_shape, levels, downs_t, strides_t, loss_fn, sr,
                 emb_width, l_bins, mu, commit, spectral, multispectral, forward_params, augment_loss=1,
                 multipliers=None, use_bottleneck=True, **params):
        super().__init__()

        self.sample_length = input_shape[0]
        self.sr = sr
        x_shape, x_channels = input_shape[:-1], input_shape[-1]
        self.x_shape = x_shape
        self.augment_loss = augment_loss

        self.downsamples = calculate_strides(strides_t, downs_t)
        self.hop_lengths = np.cumprod(self.downsamples)
        self.z_shapes = [(x_shape[0] // self.hop_lengths[level],) for level in range(levels)]
        self.levels = levels
        self.forward_params = forward_params
        self.loss_fn = loss_fn

        if multipliers is None:
            self.multipliers = [1] * levels
        else:
            assert len(multipliers) == levels, "Invalid number of multipliers"
            self.multipliers = multipliers
        def _block_kwargs(level):
            this_block_kwargs = dict(params)
            this_block_kwargs["width"] *= self.multipliers[level]
            this_block_kwargs["depth"] *= self.multipliers[level]
            return this_block_kwargs

        encoder = lambda level: Encoder(x_channels, emb_width, level + 1,
                                        downs_t[:level+1], strides_t[:level+1], **_block_kwargs(level))
        decoder = lambda level: Decoder(x_channels, emb_width, level + 1,
                                        downs_t[:level+1], strides_t[:level+1], **_block_kwargs(level))
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in range(levels):
            self.encoders.append(encoder(level))
            self.decoders.append(decoder(level))

        if use_bottleneck:
            self.bottleneck = Bottleneck(l_bins, emb_width, mu, levels)
        else:
            self.bottleneck = NoBottleneck(levels)

        self.downs_t = downs_t
        self.strides_t = strides_t
        self.l_bins = l_bins
        self.commit = commit
        self.spectral = spectral
        self.multispectral = multispectral
        self.opt_params = params

    def preprocess(self, x):
        # x: NTC [-1,1] -> NCT [-1,1]
        assert len(x.shape) == 3
        x = x.permute(0,2,1).float()
        if not x.is_cuda:
            x = x.to("cuda")
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

    def encode_vec(self, x):
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])
        return xs, x_in

    def encode_vec_with_loss(self, x):
        xs, x_in = self.encode_vec(x)
        zs, xs_quantised, commit_losses, quantiser_metrics = self.bottleneck(xs)
        return commit_losses, xs_quantised, zs, quantiser_metrics, x_in, xs

    def sample(self, n_samples):
        zs = [t.randint(0, self.l_bins, size=(n_samples, *z_shape), device='cuda') for z_shape in self.z_shapes]
        return self.decode(zs)

    def get_effects(self, pace, verbose=False):
        gain = np.random.randint(40)
        color = np.random.randint(40)
        with_reverb = bool(np.random.randint(2))
        dither = bool(np.random.randint(2))
        #flanger = bool(np.random.randint(2))
        bass_gain = np.random.randint(20)
        treble_gain = np.random.randint(20)
        if verbose:
            print("Augmented with {}".format(", ".join("{} = {}".format(k, v) for k, v in dict(
                pace=pace, gain=gain, color=color, with_reverb=with_reverb,
                dither=dither, bass_gain=bass_gain, treble_gain=treble_gain
            ).items())))

        # Define effects
        return [e for e in [
#            ["lowpass", "-1", "300"],  # apply single-pole lowpass filter
            ["tempo", str(pace)],  # reduce the speed
            ["bass", str(bass_gain)] if bass_gain > 0 else None,
            ["treble", str(treble_gain)] if treble_gain > 0 else None,
            ["overdrive", str(gain), str(color)] if gain > 0 else None,
            #["flanger"] if flanger else None,
            ["reverb", "-w"] if with_reverb else None,
            ["dither"] if dither else None,
        ] if e is not None]

    def augment(self, signal, verbose=False):
        reduce_size = lambda x: ((x[0] + x[1])/2).unsqueeze(0) if x.shape[0] > 1 else x
        pace = np.random.uniform(0.5, 1.5)

        # Apply effects
        aug_signal, srs = zip(*[apply_effects_tensor(s, self.sr, self.get_effects(pace, verbose=verbose)) for s in signal])
        assert all(s == self.sr for s in srs)
        aug_signal = t.stack(list(map(reduce_size, aug_signal)))
        return aug_signal, pace

    def suit_pace(self, signal: torch.Tensor, pace: float, shape: tuple) -> torch.Tensor:
        if pace > 1.:  # we are now upsampling (tempo was increased)
            #print("upsampling")
            positions = torch.round(torch.arange(shape[2]) * 1/pace).type(torch.LongTensor).to("cuda")
            positions = torch.clamp(positions, max=signal.shape[2] - 1, min=0)
            #positions[-1] = positions[-1] - 1 if positions[-1] >= signal.shape[1] else positions[-1
            #positions[-1] = positions[-2] - 1 if positions[-1] >= signal.shape[1] else positions[-1]
            out = signal[:, :, positions]
            #print("\n".join(list(map(repr, [signal, out, positions]))))
        else:  # we are downsampling
            #print("downsampling")
            out = torch.zeros(shape).type_as(signal).to("cuda")
            positions = torch.round(torch.arange(signal.shape[2]) * pace).type(torch.LongTensor).to("cuda")

            positions = torch.clamp(positions, max=out.shape[2] - 1, min=0)
            #positions[-1] = positions[-1] - 1 if positions[-1] >= out.shape[1] else positions[-1]
            #positions[-2] = positions[-2] - 1 if positions[-2] >= out.shape[1] else positions[-2]
            #print("\n".join(list(map(repr, [signal, out, positions]))))
            #print(torch.min(positions), torch.max(positions))
            out[:, :, positions] = signal
        return out

    def augmentation_is_close(self, signal, enc_disc_signal, enc_signal, verbose=False):
        lvls = len(enc_disc_signal)
        aug_signal, pace = self.augment(signal.permute(0, 2, 1).to("cpu"), verbose=verbose)

        enc_aug_signal, _ = self.encode_vec(aug_signal.permute(0, 2, 1))
        st_enc_aug_signal = [self.suit_pace(eas, pace, es.shape) for eas, es in zip(enc_aug_signal, enc_signal)]
        aug_loss = sum([t.mean((seas - es) ** 2) for seas, es in zip(st_enc_aug_signal, enc_signal)]) / lvls

        enc_aug_disc_signal, _, commit_losses, _ = self.bottleneck(st_enc_aug_signal)
        aug_acc = sum([t.mean((es == eas).type(torch.float)) for es, eas in zip(enc_disc_signal, enc_aug_disc_signal)]) / lvls

        return aug_loss, aug_acc, aug_signal, commit_losses
        #print(torch.min(stretched_enc_aug[0]), torch.max(stretched_enc_aug[0]))
        #print("\n".join(list(map(repr, [signal, ax, stretched_enc_aug[0], encoded_signal[0]]))))
        #print("\n".join(list(map(repr, [encoded_aug_signal[0], decoded_aug_signal[0], decoded_signal[0]]))))

    def forward(self, x):
        hps = self.forward_params
        loss_fn = self.loss_fn
        metrics = {}

        N = x.shape[0]

        # Encode/Decode
        commit_losses, xs_quantised, zs, quantiser_metrics, x_in, encoded_signal = self.encode_vec_with_loss(x)

        aug_loss, aug_acc, _, aug_commit_losses = self.augmentation_is_close(x, zs, encoded_signal) \
            if self.augment_loss > 0 else ([0], [0], 0, [0])

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
            this_recons_loss = _loss_fn(loss_fn, x_target, x_out, hps) if loss_fn != 0 else t.zeros(()).to(x.device)
            this_spec_loss = _spectral_loss(x_target, x_out, hps)
            this_multispec_loss = _multispectral_loss(x_target, x_out, hps)
            metrics[f'recons_loss_l{level + 1}'] = this_recons_loss
            metrics[f'spectral_loss_l{level + 1}'] = this_spec_loss
            metrics[f'multispectral_loss_l{level + 1}'] = this_multispec_loss
            recons_loss += this_recons_loss
            spec_loss += this_spec_loss
            multispec_loss += this_multispec_loss

        aug_commit_losses = sum(aug_commit_losses)
        commit_loss = sum(commit_losses) + aug_commit_losses
        loss = recons_loss + self.spectral * spec_loss + self.multispectral * multispec_loss +\
               self.commit * commit_loss + self.augment_loss * aug_loss

        with t.no_grad():
            sc = t.mean(spectral_convergence(x_target, x_out, hps))
            l2_loss = _loss_fn("l2", x_target, x_out, hps)
            l1_loss = _loss_fn("l1", x_target, x_out, hps)
            linf_loss = _loss_fn("linf", x_target, x_out, hps)

        quantiser_metrics = average_metrics(quantiser_metrics)

        metrics.update(dict(
            aug_loss=aug_loss,
            aug_acc=aug_acc,
            recons_loss=recons_loss,
            spectral_loss=spec_loss,
            multispectral_loss=multispec_loss,
            spectral_convergence=sc,
            l2_loss=l2_loss,
            l1_loss=l1_loss,
            linf_loss=linf_loss,
            commit_loss=commit_loss,
            aug_commit_losses=aug_commit_losses,
            **quantiser_metrics))

        for key, val in metrics.items():
            metrics[key] = val.detach()

        return x_out, loss, metrics

    def training_step(self, batch, batch_idx):
        x_out, loss, metrics = self(batch)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for name, val in metrics.items():
            self.log(name, val,  on_step=True, on_epoch=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x_out, loss, metrics = self(batch)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_out, loss, metrics = self(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for name, val in metrics.items():
            self.log("val_" + name, val,  on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def configure_optimizers(self):
        opt, sched, scalar = get_optimizer(self, **self.opt_params)
        self.scalar = scalar
        return [opt], [sched]
