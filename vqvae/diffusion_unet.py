import torch
from pytorch_lightning import LightningModule

from vqvae.modules.backbone import WavAutoEncoder
from vqvae.model import WavCompressor
from optimization.opt_maker import get_optimizer
from vqvae.modules.diffusion import Diffusion


class DiffusionUnet(LightningModule):
    def __init__(self, autenc_params, vae: WavCompressor, diff_params, log_sample_bs, encode_chunks, prep_level,
                 opt_params):
        super().__init__()
        self.opt_params = opt_params
        self.diffusion = Diffusion(**diff_params)
        self.autenc = WavAutoEncoder(**autenc_params)
        self.vae = vae
        self.log_sample_bs = log_sample_bs
        self.encode_chunks = encode_chunks
        self.prep_level = prep_level

    def __str__(self):
        return f"Diffusion model on {self.vae}"

    @torch.no_grad()
    def no_grad_forward(self, x_in):
        return self(x_in)

    def forward(self, x_in, with_metrics=False):
        x_predicted, _, _, metrics = self.autenc.forward(x_in)
        if with_metrics:
            return x_predicted, metrics
        return x_predicted

    def preprocess(self, sound):
        return self.vae.encode(sound, self.prep_level, self.prep_level + 1, bs_chunks=self.prep_chunks)[0]

    def postprocess(self, latent_sound):
        return self.vae.decode(latent_sound, self.prep_level, self.prep_level + 1, bs_chunks=self.prep_chunks)

    def noising(self, x):
        t = self.diffusion.sample_timesteps(x.shape[0])
        x_noised = self.diffusion.noise_images(x, t)
        x_noised_target = self.diffusion.noise_images(x, t - 1)
        return x_noised, x_noised_target, t

    def sample(self, bs):
        latent_samples = self.diffusion.sample(self, bs)
        samples = self.postprocess(latent_samples)
        return samples

    def calc_loss(self, x_predicted, x_noised_target, metrics):
        mse = torch.mean((x_predicted - x_noised_target)**2)
        return mse, metrics

    def evaluation_step(self, batch, batch_idx, phase="train"):
        x_in = self.generator.preprocess(batch)
        x_noised, x_noised_target, t = self.noising(x_in)

        x_predicted, metrics = self(x_noised, t, with_metrics=True)
        loss, metrics = self.calc_loss(x_predicted, x_noised_target, metrics)

        sounds_dict = dict(
            x_in=x_in, x_noised_target=x_noised_target,
            x_predicted=x_predicted, x_noised=x_noised,
        )
        self.log_metrics_and_samples(loss, metrics, sounds_dict, t, batch_idx, phase)
        return loss

    # lightning train boilerplate

    def training_step(self, batch, batch_idx):
        return self.evaluation_step(batch[0], batch_idx, "train")

    def test_step(self, batch, batch_idx):
        return self.evaluation_step(batch[0], batch_idx, "test")

    def validation_step(self, batch, batch_idx):
        return self.evaluation_step(batch[0], batch_idx, "val")

    def configure_optimizers(self):
        gopt, gsched = get_optimizer(self.generator, **self.opt_params)
        return [gopt], [gsched]

    # metrics & logging

    def log_metrics_and_samples(self, loss, metrics, sounds_dict, t, batch_idx, phase):
        prefix = phase + "_" if phase != "train" else ""
        self.log(prefix + "loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        for name, val in metrics.items():
            self.log(prefix + name, val, on_step=True, on_epoch=True, logger=True, sync_dist=True)

        if not (batch_idx % self.log_interval == 0 and self.local_rank == 0 and
                (phase == "train" or not self.skip_valid_logs)):
            return  # log samples once per interval

        samples = self.sample(self.log_sample_bs)
        self.autenc.plot_spec_as.plot_spec_as(samples, lambda i: f"generated_samples/{i}", prefix)
        self.autenc.audio_logger.log_sounds(samples, lambda i: f"generated_specs/{i}", prefix)
        for name, latent_sound in sounds_dict.items():
            sound = self.postprocess(latent_sound)
            self.autenc.plot_spec_as.plot_spec_as(sound, lambda i: f"spec_{i}/{name}", prefix)
            self.autenc.audio_logger.log_sounds(sound, lambda i: f"sample_{i}/{name}", prefix)
