import torch
import torch.nn as nn
from utils.misc import default, bms


class DiffusionStats(nn.Module):
    def __init__(self, steps, intervals, cond, momentum, renormalize_loss):
        super().__init__()
        self.cond = [cond]
        self.renormalize_loss = renormalize_loss
        self.steps = steps
        self.momentum = momentum
        self.intervals = intervals
        self.int_names = ["total" if iv == (1, self.steps) else iv[0] if iv[0] == iv[1]
                          else f"{iv[0]}-{iv[1]}" for iv in intervals]
        self.stats = dict()

    def bnorm(self, x):
        return torch.mean(torch.sqrt(bms(x)))

    def append_t_metric(self, key, value, t):
        if key not in self.stats:
            self.stats[key] = torch.zeros((self.steps,), dtype=torch.float, device=value.device) + torch.nan
        if torch.isnan(self.stats[key][t]):
            self.stats[key][t] = value
        else:
            self.stats[key][t] = self.momentum * self.stats[key][t] + (1 - self.momentum) * value

    @torch.no_grad()
    def aggregate(self, t, metrics):
        for key, value in metrics.items():
            for b in range(len(t)):
                self.append_t_metric(key, value[b], t[b])

    @torch.no_grad()
    def get_aggr(self):
        metrics = {}
        for intv, i_name in zip(self.intervals, self.int_names):
            for k, v in self.stats.items():
                mean = torch.nanmean(self.stats[k][intv[0] - 1:intv[1]])
                if not torch.isnan(mean):
                    metrics[f"{k}/{i_name}"] = mean
        return metrics

    def get_loss_weights(self, t, name):
        dist = self.stats.get(name, torch.ones_like(t, device=t.device))
        dist = torch.nanmean(dist) / torch.cat([dist[ti] for ti in t])
        return torch.nan_to_num(dist, nan=1.)

    def get_sample_info(self, i, t=None, context=None, time=None):
        full_name = []
        if t is not None:
            full_name.append(f"t={t[i]}")
        if context is not None:
            genres = ",".join([self.cond[0].genre_names[gid] for gid in context[i].genres])
            full_name.append(f"({genres})")
        return "_".join(full_name)

    def residual_metrics(self, pred, target, name, t):
        mse = bms(pred - target)
        rmse = mse**.5
        p_norm = bms(pred)**.5
        t_var = bms(target)
        t_norm = t_var**.5
        r_squared = 1 - mse/t_var
        metrics = dict(mse=mse, rmse=rmse, norm=t_norm, pred_norm=p_norm, r_squared=r_squared)

        loss = mse if self.renormalize_loss else mse * self.get_loss_weights(t, f"{name}/mse")
        return torch.mean(loss), {f"{name}/{k}": v for k, v in metrics.items()}
