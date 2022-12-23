import torch
import torch.nn as nn


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

    def forward(self, x):
        dist_params = self.to_params(x.permute(0, 2, 1)).permute(0, 2, 1).reshape(x.shape[0], 2, *x.shape[1:])
        mu, logvar = dist_params[:, 0], dist_params[:, 1]
        var = torch.exp(logvar)

        z = mu + var * torch.randn_like(mu)

        kl_loss = self.get_kl_div(mu, var, logvar)
        metrics = {"vae_sigma": torch.mean(var), "vae_mu": torch.mean(mu)}
        return z, kl_loss, metrics

    def encode(self, x):
        z, _, _ = self(x)
        return z

    def decode(self, x):
        return x

