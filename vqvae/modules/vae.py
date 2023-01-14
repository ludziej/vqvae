import torch
import torch.nn as nn
from utils.misc import default


class VAEBottleneck(nn.Module):
    def __init__(self, emb_width, levels, kl_div_weight=1):
        super().__init__()
        self.levels = levels
        self.level_blocks = nn.ModuleList([VAEBottleneckBlock(emb_width, lvl, kl_div_weight)
                                           for lvl in range(self.levels)])

    def forward(self, xs):
        assert len(xs) == self.levels
        outs = [block(x) for level, (x, block) in enumerate(zip(xs, self.level_blocks))]
        zs, losses, metrics = zip(*outs)
        return zs, zs, losses, 0, metrics

    def encode(self, xs, var_temp=None):
        assert len(xs) == self.levels
        x_coded = [block.encode(x, var_temp=var_temp) for x, block in zip(xs, self.level_blocks)]
        return x_coded

    def decode(self, xs, start_level=0, end_level=None):
        end_level = default(end_level, self.levels)
        x_decoded = [self.level_blocks[i].decode(xs[xi]) for xi, i in enumerate(range(start_level, end_level))]
        return x_decoded


class VAEBottleneckBlock(nn.Module):
    def __init__(self, emb_width, level, kl_div_weight):
        super().__init__()
        self.level = level
        self.emb_width = emb_width
        self.kl_div_weight = kl_div_weight
        self.to_params = nn.Linear(emb_width, 2 * emb_width)

    def get_kl_div(self, mu, var, logvar):
        kld_element = -0.5 * (1 + logvar - mu ** 2 - var) * self.kl_div_weight
        kld_loss = torch.mean(torch.sum(kld_element, dim=-2))
        return kld_loss

    def forward(self, x, var_temp=None):
        dist_params = self.to_params(x.permute(0, 2, 1)).permute(0, 2, 1).reshape(x.shape[0], 2, *x.shape[1:])
        mu, logvar = dist_params[:, 0], dist_params[:, 1]
        var = torch.exp(logvar)
        if var_temp is not None:
            var *= var_temp
            logvar = torch.log(var)

        z = mu + var * torch.randn_like(mu)

        kl_loss = self.get_kl_div(mu, var, logvar)
        metrics = {"vae_sigma": torch.mean(var), "vae_mu": torch.mean(mu)}
        return z, kl_loss, metrics

    def encode(self, x, var_temp=None):
        z, _, _ = self(x, var_temp=var_temp)
        return z

    def decode(self, x, start_level=0, end_level=None):
        return x

