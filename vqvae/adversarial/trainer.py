from collections import ChainMap

import torch
from torch import nn as nn

from vqvae.adversarial.model import JoinedDiscriminator, FFTDiscriminator, WavDiscriminator


def get_discriminator(with_discriminator, type, **params):
    if not with_discriminator:
        return None
    if type == "joined":
        return JoinedDiscriminator(**params)
    elif type == "mel":
        return FFTDiscriminator(**params)
    elif type == "wav":
        return WavDiscriminator(**params)
    raise Exception("Not Implemented")


class AdversarialTrainer(nn.Module):
    def __init__(self, gan_loss_weight, adv_latency, levels, with_discriminator, **params):
        super().__init__()
        self.with_discriminator = with_discriminator
        self.levels = levels
        self.gan_loss_weight = gan_loss_weight
        self.adv_latency = adv_latency
        self.discriminator = get_discriminator(with_discriminator=with_discriminator, **params)

    def forward(self, x_in, gen_out, optimize_generator=False):
        bs = x_in.shape[0]
        gen_out = torch.cat(gen_out, dim=0)
        gen_out = gen_out if optimize_generator else gen_out.detach()
        x_in = x_in[0:0] if optimize_generator else x_in  # no need to forward on input optimizing generator
        in_batch = torch.cat([gen_out, x_in])

        y_true = torch.cat([torch.zeros(len(gen_out)), torch.ones(len(x_in))]).to(x_in.device).long()
        y_opt = 1 - y_true if optimize_generator else y_true

        calced_stats = self.discriminator.calculate_loss(in_batch, y_opt, balance=not optimize_generator)
        loss = sum([x.loss for x in calced_stats])
        params = dict(optimize_generator=optimize_generator, bs=bs, y=y_true)
        metrics = ChainMap(*[self.discriminator_metrics(**params, **stats._asdict()) for stats in calced_stats])
        return loss, dict(metrics)

    def training_step(self, metrics, optimizer_idx, x_in, x_outs, batch_idx, current_epoch):
        if not self.with_discriminator or (self.adv_latency > batch_idx and current_epoch == 0):
            return 0
        optimize_generator = optimizer_idx == 0
        gan_loss, adv_metrics = self.forward(x_in, x_outs, optimize_generator=optimize_generator)
        metrics.update(adv_metrics)
        return gan_loss * self.gan_loss_weight

    def discriminator_metrics(self, optimize_generator, loss, pred, acc, bs, cls, y, loss_ew, name):
        prefix = f"{name}_{('generator' if optimize_generator else 'discriminator')}"
        metrics = {prefix + "_loss/total": loss}
        loss_real = None if optimize_generator else torch.mean(loss_ew[-bs:])
        loss_ew = loss_ew[:self.levels * bs].reshape(self.levels, bs)
        for level, lvl_loss in enumerate(loss_ew):
            metrics[f"{prefix}_loss/lvl_{level + 1}"] = torch.mean(lvl_loss) if optimize_generator else \
                (torch.mean(lvl_loss) + torch.mean(loss_real))/2

        if not optimize_generator:  # log per level acc stats
            metrics[f"{prefix}_acc/balanced"] = acc
            error = (y == cls).reshape(self.levels + 1, bs).float()
            real_acc = torch.mean(error[-1])  # accuracy on real input
            metrics[f"{prefix}_acc/on_real"] = real_acc
            for i in range(self.levels):
                fake_acc = torch.mean(error[i])  # accuracy on fake input on given level
                metrics[f"{prefix}_acc/lvl_{i + 1}"] = (real_acc + fake_acc) / 2  # average with real to balance
                metrics[f"{prefix}_acc/on_fake_lvl_{i + 1}"] = fake_acc
        return metrics
