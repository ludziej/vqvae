import torch
import torch.nn as nn
from tqdm import tqdm
import logging
from utils.misc import default


# code from https://github.com/dome272/Diffusion-Models-pytorch/blob/main/ddpm.py
class Diffusion(nn.Module):
    def __init__(self, emb_width, renormalize_sampling=False, noise_steps=1000, beta_start=1e-4, beta_end=0.02):
        super(Diffusion, self).__init__()
        self.emb_width = emb_width
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.renormalize_sampling = renormalize_sampling
        self.const_x0_post = False

        self.register_buffer('beta', self.prepare_noise_schedule())
        self.register_buffer('alpha', 1. - self.beta)
        self.register_buffer('alpha_hat', torch.cumprod(self.alpha, dim=0))

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

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

    def denoise_step(self, x, predicted_noise, t, add_noise=False):
        alpha, alpha_hat, beta = [self.extract(d, t) for d in [self.alpha, self.alpha_hat, self.beta]]

        posterior = beta * (1. - alpha_hat/alpha) / (1. - alpha_hat) if self.const_x0_post else beta
        noise = torch.randn_like(x) if add_noise else torch.zeros_like(x)
        x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) \
            + torch.sqrt(posterior) * noise
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
                predicted_noise = model(x, t, **context_args)
                x = self.denoise_step(x, predicted_noise[level], t, add_noise=i > 1)
                norm = torch.sqrt(torch.mean(x**2))
                if self.renormalize_sampling or norm >= 100:
                    x *= 1 / norm
        model.train(was_training)
        return x
