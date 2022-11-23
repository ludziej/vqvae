from hparams.compressor.misc import dirs_config
from hparams.parser import Hparams

default_vqvae_opt_hparams = Hparams(
    chunk_timeout=2,
    chunk_another_thread=False,
    shuffle_data=True,
    use_audiofile=False,
    epochs=10000,
    lr=0.0003,
    clip=1.0,
    beta1=0.9,
    beta2=0.999,
    ignore_grad_norm=0,
    weight_decay=1e-6,
    eps=1e-08,
    rms_normalize_sound=True,
    rms_normalize_level=-13,
    prenorm_normalisation=None,
    prenorm_loss_weight=1,
    lr_warmup=100.0,
    lr_decay=3000,
    lr_gamma=0.3,
    lr_scale=1.0,
    lr_use_linear_decay=False,
    lr_start_linear_decay=0,
    ckpt_freq=10,
    band_est_dur=1000,
    #ckpt_name='model_{epoch}_{loss_step:.2f}',
    ckpt_name='last_model',
    **dirs_config.__dict__,
)
big_vqvae_opt_hparams = default_vqvae_opt_hparams.update(
    weight_decay=1e-4,
)
small_vqvae_opt_hparams = default_vqvae_opt_hparams.update(
    weight_decay=0.0,
)
