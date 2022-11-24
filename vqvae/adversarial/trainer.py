from collections import ChainMap

import torch
from torch import nn as nn

from vqvae.adversarial.model import JoinedDiscriminator, FFTDiscriminator, WavDiscriminator


def get_discriminator(with_discriminator, type, **params):
    if not with_discriminator:
        return None
    if type == "joined":
        return JoinedDiscriminator(**params)
    elif type in ("mel", "fft", "cqt"):
        return FFTDiscriminator(prep_type=type, **params)
    elif type == "wav":
        return WavDiscriminator(**params)
    raise Exception(f"Unknown discriminator type = {type}")


class AdversarialTrainer(nn.Module):
    def __init__(self, gan_loss_weight, gan_loss_warmup, adv_latency, levels, with_discriminator,
                 disc_loss_weight, disc_use_freq, stop_disc_train_after, classify_each_level, **params):
        super().__init__()
        self.disc_use_freq = disc_use_freq
        self.with_discriminator = with_discriminator
        self.levels = levels
        self.gan_loss_weight = gan_loss_weight
        self.gan_loss_warmup = gan_loss_warmup
        self.adv_latency = adv_latency
        self.disc_loss_weight = disc_loss_weight
        self.stop_disc_train_after = stop_disc_train_after
        self.discriminator = get_discriminator(with_discriminator=with_discriminator, levels=levels,
                                               classify_each_level=classify_each_level, **params)

    def forward(self, x_in, gen_out, optimize_generator=False):
        bs = x_in.shape[0]
        gen_out = torch.cat(gen_out, dim=0)
        gen_out = gen_out if optimize_generator else gen_out.detach()
        x_in = x_in[0:0] if optimize_generator else x_in  # no need to forward on input optimizing generator
        in_batch = torch.cat([gen_out, x_in])

        origins = sum([bs*[i + 1] for i in range(self.levels)], []) + len(x_in)*[0]
        y_true = torch.tensor(origins, device=gen_out.device, dtype=torch.long)
        y_opt = torch.zeros(y_true.shape, device=y_true.device).long() if optimize_generator else y_true

        calced_stats = self.discriminator.calculate_loss(in_batch, y_opt, balance=not optimize_generator)
        loss = sum([x.loss for x in calced_stats])
        params = dict(optimize_generator=optimize_generator, bs=bs, y=y_true)
        metrics = ChainMap(*[self.discriminator_metrics(**params, **stats._asdict()) for stats in calced_stats])
        return loss, dict(metrics)

    def is_used(self, batch_idx, current_epoch, optimize_generator):
        return self.with_discriminator and (self.adv_latency <= batch_idx or current_epoch != 0) and \
               (batch_idx % self.disc_use_freq == 0 or optimize_generator) and \
               not (batch_idx > self.stop_disc_train_after and not optimize_generator)

    def training_step(self, metrics, optimize_generator, x_in, x_outs, batch_idx, current_epoch):
        if not self.is_used(batch_idx, current_epoch, optimize_generator):
            return 0
        gan_loss, adv_metrics = self.forward(x_in, x_outs, optimize_generator=optimize_generator)
        metrics.update(adv_metrics)
        return self.adjust_adv_lr(gan_loss, batch_idx, metrics, optimize_generator)

    def adjust_adv_lr(self, gan_loss, batch_idx, metrics, optimize_generator):
        weight_modifier = self.gan_loss_weight
        if optimize_generator:  # warmup
            weight_modifier *= min(1, (batch_idx - self.adv_latency) / self.gan_loss_warmup)
            metrics["gan_loss_weight"] = torch.tensor(weight_modifier, dtype=torch.float)
        else:  # discriminator adjustment
            weight_modifier *= self.disc_loss_weight
        return gan_loss * weight_modifier

    def discriminator_metrics(self, optimize_generator, loss, pred, acc, bs, cls, y, loss_ew, name):
        prefix = f"{name}_{('generator' if optimize_generator else 'discriminator')}"
        metrics = {prefix + "_loss/total": loss}
        loss_real = None if optimize_generator else torch.mean(loss_ew[-bs:])
        loss_ew = loss_ew[:self.levels * bs].reshape(self.levels, bs)
        for level, lvl_loss in enumerate(loss_ew):
            metrics[f"{prefix}_loss/lvl_{level + 1}"] = torch.mean(lvl_loss) if optimize_generator else \
                (torch.mean(lvl_loss) + torch.mean(loss_real))/2

        metrics[f"{name}_discriminator_acc/balanced"] = acc
        correct = (y == cls).reshape(-1, bs).float()
        if not optimize_generator:
            real_acc = torch.mean(correct[-1])  # accuracy on real input
            metrics[f"{name}_discriminator_acc/on_real"] = real_acc
        for i in range(self.levels):
            fake_acc = torch.mean(correct[i])  # accuracy on fake input on given level
            metrics[f"discriminator_acc/on_fake_lvl_{i + 1}"] = fake_acc
            if not optimize_generator:
                metrics[f"{name}_discriminator_acc/lvl_{i + 1}"] = (real_acc + fake_acc) / 2  # average with real to balance
        return metrics
