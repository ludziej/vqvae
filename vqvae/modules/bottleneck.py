import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F


class BottleneckBlock(nn.Module):
    def __init__(self, k_bins, emb_width, mu, norm_before_vqvae, fixed_commit):
        super().__init__()
        self.k_bins = k_bins
        self.fixed_commit = fixed_commit
        self.emb_width = emb_width
        self.mu = mu
        self.norm_before_vqvae = norm_before_vqvae
        self.threshold = 1.0
        self.reset_k()

    def reset_k(self):
        self.init = False
        if self.norm_before_vqvae:
            self.register_buffer('k', t.rand(self.k_bins, self.emb_width) - 0.5)
            self.k = t.nn.functional.normalize(self.k)
        else:
            self.register_buffer('k', t.zeros(self.k_bins, self.emb_width))

        self.register_buffer('k_sum', t.zeros(self.k_bins, self.emb_width))
        self.register_buffer('k_elem', t.ones(self.k_bins, device=self.k.device))

    def _tile(self, x):
        d, ew = x.shape
        if d < self.k_bins:
            n_repeats = (self.k_bins + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + t.randn_like(x) * std
        return x

    def init_k(self, x):
        emb_width, k_bins = self.emb_width, self.k_bins
        self.init = True
        # init k_w using random vectors from x
        y = self._tile(x)
        _k_rand = y[t.randperm(y.shape[0])][:k_bins]
        self.k = _k_rand
        assert self.k.shape == (k_bins, emb_width)
        self.k_sum = self.k.detach()
        self.k_elem = t.ones(k_bins, device=self.k.device)

    def update_k(self, x, x_l):
        mu, emb_width, k_bins = self.mu, self.emb_width, self.k_bins
        with t.no_grad():
            # Calculate new centres
            x_l_onehot = t.zeros(k_bins, x.shape[0], device=x.device)  # k_bins, N * L
            x_l_onehot.scatter_(0, x_l.view(1, x.shape[0]), 1)

            _k_sum = t.matmul(x_l_onehot, x)  # k_bins, w
            _k_elem = x_l_onehot.sum(dim=-1)  # k_bins
            y = self._tile(x)
            _k_rand = y[t.randperm(y.shape[0])][:k_bins]

            # Update centres
            old_k = self.k.detach()
            self.k_sum = mu * self.k_sum + (1. - mu) * _k_sum  # w, k_bins
            self.k_elem = mu * self.k_elem + (1. - mu) * _k_elem  # k_bins
            usage = (self.k_elem.view(k_bins, 1) >= self.threshold).float()
            self.k = usage * (self.k_sum.view(k_bins, emb_width) / self.k_elem.view(k_bins, 1)) + (1 - usage) * _k_rand
            if self.norm_before_vqvae:
                self.k = t.nn.functional.normalize(self.k)
            _k_prob = _k_elem / t.sum(_k_elem)  # x_l_onehot.mean(dim=-1)  # prob of each bin
            entropy = -t.sum(_k_prob * t.log(_k_prob + 1e-8))  # entropy ie how diverse
            used_curr = (_k_elem >= self.threshold).sum().float()
            usage = t.sum(usage)
            dk = t.norm(self.k.detach() - old_k) / np.sqrt(np.prod(old_k.shape))
        return dict(entropy=entropy,
                    used_curr=used_curr,
                    usage=usage,
                    dk=dk)

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  # x_en = (N * L, w), k_j = (w, k_bins)

        prenorm_vec = t.mean(t.norm(x.detach(), dim=1))
        if x.shape[-1] == self.emb_width:
            prenorm = t.norm(x - t.mean(x)) / np.sqrt(np.prod(x.shape))
        elif x.shape[-1] == 2 * self.emb_width:
            x1, x2 = x[...,:self.emb_width], x[...,self.emb_width:]
            prenorm = (t.norm(x1 - t.mean(x1)) / np.sqrt(np.prod(x1.shape))) + (t.norm(x2 - t.mean(x2)) / np.sqrt(np.prod(x2.shape)))

            # Normalise
            x = x1 + x2
        else:
            assert False, f"Expected {x.shape[-1]} to be (1 or 2) * {self.emb_width}"
        if self.norm_before_vqvae:
            x = t.nn.functional.normalize(x)
        return x, prenorm, prenorm_vec

    def postprocess(self, x_l, x_d, x_shape):
        # [NT, C] -> NTC -> NCT
        N, T = x_shape
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        x_l = x_l.view(N, T)
        return x_l, x_d

    def quantise(self, x):
        # Calculate latent code x_l
        k_w = self.k.detach().t()
        distance = t.sum(x ** 2, dim=-1, keepdim=True) - 2 * t.matmul(x, k_w) + t.sum(k_w ** 2, dim=0,
                                                                                            keepdim=True)  # (N * L, b)
        min_distance, x_l = t.min(distance, dim=-1)
        fit = t.mean(min_distance)
        return x_l, fit

    def dequantise(self, x_l):
        x = F.embedding(x_l, self.k.detach())
        return x

    def encode(self, x):
        N, width, T = x.shape

        # Preprocess.
        x, prenorm, prenorm_vec = self.preprocess(x)

        # Quantise
        x_l, fit = self.quantise(x)

        # Postprocess.
        x_l = x_l.view(N, T)
        return x_l

    def decode(self, x_l):
        N, T = x_l.shape
        width = self.emb_width

        # Dequantise
        x_d = self.dequantise(x_l)

        # Postprocess
        x_d = x_d.view(N, T, width).permute(0, 2, 1).contiguous()
        return x_d

    def forward(self, x, update_k=True):
        N, width, T = x.shape

        # Preprocess
        x, prenorm, prenorm_vec = self.preprocess(x)

        # Init k if not inited
        if update_k and not self.init:
            self.init_k(x)

        # Quantise and dequantise through bottleneck
        x_l, fit = self.quantise(x)
        x_d = self.dequantise(x_l)

        # Update embeddings
        if update_k:
            update_metrics = self.update_k(x, x_l)
        else:
            update_metrics = {}

        # Loss
        commit_loss = t.mean(t.norm(x_d.detach() - x, dim=1)) if self.fixed_commit else\
            t.norm(x_d.detach() - x) ** 2 / np.prod(x.shape)

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_l, x_d = self.postprocess(x_l, x_d, (N,T))
        return x_l, x_d, commit_loss, prenorm, dict(fit=fit,
                                                    pn=prenorm,
                                                    pn_vec=prenorm_vec,
                                                    **update_metrics)


class Bottleneck(nn.Module):
    def __init__(self, l_bins, emb_width, mu, levels, norm_before_vqvae, fixed_commit):
        super().__init__()
        self.levels = levels
        self.level_blocks = nn.ModuleList([BottleneckBlock(l_bins, emb_width, mu, norm_before_vqvae,
                                                           fixed_commit)
                                           for level in range(self.levels)])

    def encode(self, xs):
        zs = [level_block.encode(x) for (level_block, x) in zip(self.level_blocks, xs)]
        return zs

    def decode(self, zs, start_level=0, end_level=None):
        if end_level is None:
            end_level = self.levels
        xs_quantised = [level_block.decode(z) for (level_block, z) in zip(self.level_blocks[start_level:end_level], zs)]
        return xs_quantised

    def forward(self, xs):
        zs, xs_quantised, commit_losses, prenorms, metrics = [], [], [], [], []
        for level in range(self.levels):
            level_block = self.level_blocks[level]
            x = xs[level]
            z, x_quantised, commit_loss, prenorm, metric = level_block(x, update_k=self.training)
            zs.append(z)
            if not self.training:
                # Be extra paranoid and make sure the encoder weights can't
                # change from straight-through estimator
                x_quantised = x_quantised.detach()
            xs_quantised.append(x_quantised)
            prenorms.append(prenorm)
            commit_losses.append(commit_loss)
            if self.training:
                metrics.append(metric)
        return zs, xs_quantised, commit_losses, prenorms, metrics

class NoBottleneckBlock(nn.Module):
    def restore_k(self):
        pass

class NoBottleneck(nn.Module):
    def __init__(self, levels):
        super().__init__()
        self.level_blocks = nn.ModuleList()
        self.levels = levels
        for level in range(levels):
            self.level_blocks.append(NoBottleneckBlock())

    def encode(self, xs):
        return xs

    def decode(self, zs, start_level=0, end_level=None):
        if end_level is None:
            end_level = self.levels
        return zs

    def forward(self, xs):
        zero = t.zeros(()).cuda()
        commit_losses = [zero for _ in range(self.levels)]
        metrics = [dict(entropy=zero, usage=zero, used_curr=zero, pn=zero, dk=zero) for _ in range(self.levels)]
        return xs, xs, commit_losses, metrics

if __name__ == '__main__':
    from jukebox.utils.dist_utils import setup_dist_from_mpi
    rank, local_rank, device = setup_dist_from_mpi(port=29600)
    bottleneck = Bottleneck(256, 64, 0.99, 2).to(device)
    bottleneck.check()
