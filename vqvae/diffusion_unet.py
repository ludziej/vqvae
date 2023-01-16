import torch
from pytorch_lightning import LightningModule

from vqvae.modules.backbone import WavAutoEncoder
from vqvae.model import WavCompressor
from vqvae.modules.diffusion import Diffusion
from optimization.opt_maker import get_optimizer
from utils.misc import default


class DiffusionUnet(LightningModule):
    def __init__(self, autenc_params, preprocessing: WavCompressor, diff_params, log_sample_bs, prep_chunks,
                 prep_level, opt_params, logger, log_interval, **params):
        super().__init__()
        self.log_sample_bs = log_sample_bs
        self.prep_chunks = prep_chunks
        self.opt_params = opt_params
        self.prep_level = prep_level
        self.my_logger = logger
        self.preprocessing = preprocessing
        self.log_interval = log_interval
        self.skip_valid_logs = True
        self.diffusion = Diffusion(emb_width=self.preprocessing.emb_width, **diff_params)
        self.autenc = WavAutoEncoder(sr=self.preprocessing.sr,
                                     **autenc_params, input_channels=self.preprocessing.emb_width, base_model=self)

        for param in self.preprocessing.parameters():
            param.requires_grad = False

    def __str__(self):
        return f"Diffusion model on {self.preprocessing}"

    @torch.no_grad()
    def no_grad_forward(self, x_in):
        return self(x_in)

    def forward(self, x_in, with_metrics=False):
        x_predicted, _, _, metrics = self.autenc.forward(x_in)
        if with_metrics:
            return x_predicted, metrics
        return x_predicted

    @torch.no_grad()
    def preprocess(self, sound):
        return self.preprocessing.encode(sound, self.prep_level, self.prep_level + 1, bs_chunks=self.prep_chunks)[0]

    @torch.no_grad()
    def postprocess(self, latent_sound):
        decoded = self.preprocessing.decode([latent_sound], self.prep_level,
                                            self.prep_level + 1, bs_chunks=self.prep_chunks)
        return decoded.permute(0, 2, 1)

    def noising(self, x):
        t = self.diffusion.sample_timesteps(x.shape[0])
        x_noised, noise_target = self.diffusion.noise_images(x, t)
        return x_noised, noise_target, t

    @torch.no_grad()
    def sample(self, bs, length):
        latent_samples = self.diffusion.sample(self, bs, length=length)
        sample_norm = torch.mean(latent_samples**2)
        samples = self.postprocess(latent_samples)
        return samples, dict(sample_norm=sample_norm)

    def calc_loss(self, preds, target, metrics_l):
        mse = sum([torch.mean((pred - target)**2) for pred in preds])
        target_norm = torch.mean(target**2)
        for metrics, pred in zip(metrics_l, preds):
            pred_norm = torch.mean(pred**2)
            r_squared = 1 - mse/target_norm
            metrics.update(target_norm=target_norm, pred_norm=pred_norm, r_squared=r_squared)
        return mse, metrics_l

    def evaluation_step(self, batch, batch_idx, phase="train"):
        x_in = self.preprocess(batch)
        x_noised, noise_target, t = self.noising(x_in)

        noise_pred, metrics = self(x_noised, with_metrics=True)
        loss, metrics = self.calc_loss(noise_pred, noise_target, metrics)

        sounds_dict = self.get_sounds_dict(x_in, x_noised, noise_target, noise_pred, t)
        self.log_metrics_and_samples(loss, metrics, sounds_dict, t, batch_idx, phase, sample_len=x_in.shape[-1])
        return loss

    def get_sounds_dict(self, x_in, x_noised, noise_target, noise_pred, t):
        x_pred_denoised = self.diffusion.denoise_step(x_noised, noise_pred[0], t)
        x_target_denoised = self.diffusion.denoise_step(x_noised, noise_target, t)
        return dict(x_in=x_in, x_pred_denoised=x_pred_denoised,
                    x_target_denoised=x_target_denoised, x_noised=x_noised)

    # lightning train boilerplate

    def training_step(self, batch, batch_idx):
        return self.evaluation_step(batch[0], batch_idx, "train")

    def test_step(self, batch, batch_idx):
        return self.evaluation_step(batch[0], batch_idx, "test")

    def validation_step(self, batch, batch_idx):
        return self.evaluation_step(batch[0], batch_idx, "val")

    def configure_optimizers(self):
        gopt, gsched = get_optimizer(self, **self.opt_params)
        return [gopt], [gsched]

    # metrics & logging

    @torch.no_grad()
    def log_metrics_and_samples(self, loss, metrics, sounds_dict, t, batch_idx, phase, sample_len):
        prefix = phase + "_" if phase != "train" else ""
        assert len(metrics) == 1  # currently no support for multiple levels
        self.autenc.audio_logger.log_metrics({**metrics[0], "loss": loss}, prefix)

        if not (batch_idx % self.log_interval == 0 and self.local_rank == 0 and
                (phase == "train" or not self.skip_valid_logs)):
            return  # log samples once per interval

        samples, sample_metrics = self.sample(self.log_sample_bs, length=sample_len)
        self.autenc.audio_logger.log_add_metrics(sample_metrics, prefix)

        self.autenc.audio_logger.plot_spec_as(samples, lambda i: f"generated_specs/{i}", prefix)
        self.autenc.audio_logger.log_sounds(samples, lambda i: f"generated_samples/{i}", prefix)
        for name, latent_sound in sounds_dict.items():
            sound = self.postprocess(latent_sound)
            self.autenc.audio_logger.plot_spec_as(sound, lambda i: f"spec_{i}/{name}", prefix)
            self.autenc.audio_logger.log_sounds(sound, lambda i: f"sample_{i}/{name}", prefix)
