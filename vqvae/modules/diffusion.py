import torch
import torch.nn as nn
from tqdm import tqdm
import logging
from utils.misc import default
import math


def linear_noise_schedule(beta_start, beta_end, noise_steps):
    return torch.linspace(beta_start, beta_end, noise_steps)


def cosine_noise_schedule(timesteps, s):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999).type(torch.float)


# code from https://github.com/dome272/Diffusion-Models-pytorch/blob/main/ddpm.py
class Diffusion(nn.Module):
    def __init__(self, emb_width, noise_schedule, renormalize_sampling=False, noise_steps=1000, beta_start=1e-4,
                 beta_end=0.02, clip_val=1., dynamic_clipping=False, dclip_perc=0.99,
                 clip_pred=True, use_one_step=False, clip_input=False, noise_schedule_s=0.008):
        super(Diffusion, self).__init__()
        self.emb_width = emb_width
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.clip_input = clip_input
        self.clip_val = clip_val
        self.clip_pred = clip_pred
        self.use_one_step = use_one_step
        self.beta_end = beta_end
        self.noise_schedule = noise_schedule
        self.noise_schedule_s = noise_schedule_s
        self.renormalize_sampling = renormalize_sampling
        self.dynamic_clipping = dynamic_clipping
        self.dclip_perc = dclip_perc
        self.const_x0_post = False

        self.register_buffer('beta', self.prepare_noise_schedule())
        self.register_buffer('alpha', 1. - self.beta)
        self.register_buffer('alpha_hat', torch.cumprod(self.alpha, dim=0))

    def prepare_noise_schedule(self):
        if self.noise_schedule == "linear":
            return linear_noise_schedule(self.beta_start, self.beta_end, self.noise_steps)
        elif self.noise_schedule == "cosine":
            return cosine_noise_schedule(self.noise_steps, s=self.noise_schedule_s)
        else:
            raise Exception(f"Not Implemented noise schedule: {self.noise_schedule}")

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def extract(self, data, t):
        return data[t][:, None, None]

    def noise_images(self, x, t):
        alpha_hat = self.extract(self.alpha_hat, t)
        e = torch.randn_like(x)
        xt = torch.sqrt(alpha_hat) * x + torch.sqrt(1 - alpha_hat) * e
        return xt, e

    def get_x0(self, x, predicted_noise, t):
        alpha_hat = self.extract(self.alpha_hat, t)
        return 1/torch.sqrt(alpha_hat) * (x - torch.sqrt(1 - alpha_hat) * predicted_noise)

    def get_q_posterior_from_x0(self, xt, x0, t):
        alpha, alpha_hat, beta = [self.extract(d, t) for d in [self.alpha, self.alpha_hat, self.beta]]
        alpha_hat_prev = alpha_hat / alpha
        x0_coef = beta * torch.sqrt(alpha_hat_prev) / (1.0 - alpha_hat)
        xt_coef = (1.0 - alpha_hat_prev) * torch.sqrt(alpha) / (1.0 - alpha_hat)
        return x0_coef * x0 + xt_coef * xt

    def clipping(self, x):
        if not self.clip_pred:
            return x
        if self.dynamic_clipping:
            thresh_quantile = torch.quantile(torch.abs(x.reshape(x.shape[0], -1)), self.dclip_perc, dim=1)
            thresh_quantile = thresh_quantile.clamp(self.clip_val)
            x = x / thresh_quantile * self.clip_val
        return x.clamp(-self.clip_val, self.clip_val)

    def two_step_q_posterior(self, x, prediceted_noise, t):
        x0 = self.get_x0(x, prediceted_noise, t)
        x0 = self.clipping(x0)
        return self.get_q_posterior_from_x0(x, x0, t)

    def one_step_q_posterior(self, x, predicted_noise, t):
        alpha, alpha_hat, beta = [self.extract(d, t) for d in [self.alpha, self.alpha_hat, self.beta]]
        return 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise)

    def denoise_step(self, x, predicted_noise, t, add_noise=False):
        alpha, alpha_hat, beta = [self.extract(d, t) for d in [self.alpha, self.alpha_hat, self.beta]]

        q_posterior = self.one_step_q_posterior(x, predicted_noise, t) if self.use_one_step else \
            self.two_step_q_posterior(x, predicted_noise, t)

        posterior_noise = beta * (1. - alpha_hat/alpha) / (1. - alpha_hat) if self.const_x0_post else beta
        noise = torch.randn_like(x) if add_noise else torch.zeros_like(x)
        x = q_posterior + torch.sqrt(posterior_noise) * noise
        return x

    def sample(self, model, n, length, steps=None, **context_args):
        logging.info(f"Sampling {n} new images....")
        x = torch.randn((n, self.emb_width, length)).to(model.device)
        return self.denoise(x, model, steps=steps, **context_args)

    def denoise(self, x, model, steps=None, level=0, **context_args):
        was_training = self.training
        steps = default(steps, self.noise_steps)
        model.eval()
        with torch.no_grad():
            for i in tqdm(list(reversed(range(1, steps))), position=0, desc="Denoising"):
                t = (torch.ones(len(x)) * i).long().to(model.device)
                predicted_noise = model.eval_no_grad(x, t, **context_args)
                x = self.denoise_step(x, predicted_noise[level], t, add_noise=i > 1)
                norm = torch.sqrt(torch.mean(x**2))
                if self.renormalize_sampling or norm >= 100:
                    x *= 1 / norm
        model.train(was_training)
        return x
