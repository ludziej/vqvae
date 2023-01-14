from hparams.compressor.misc import dirs_config
from hparams.parser import Hparams

default_vqvae_data_hparams = Hparams(
    chunk_timeout=2,
    chunk_another_thread=False,
    shuffle_data=True,
    use_audiofile=False,
    rms_normalize_sound=True,
    rms_normalize_level=-13,
    band_est_dur=1000,
)

default_vqvae_opt_hparams = default_vqvae_data_hparams.update(
    lr=0.0003,
    beta1=0.9,
    beta2=0.999,
    weight_decay=1e-6,
    eps=1e-08,
    lr_warmup=100.0,
    lr_decay=3000,
    lr_gamma=0.3,
    lr_scale=1.0,
    lr_use_linear_decay=False,
    lr_start_linear_decay=0,
    **dirs_config.__dict__,
)
big_vqvae_opt_hparams = default_vqvae_opt_hparams.update(
    weight_decay=1e-4,
)
small_vqvae_opt_hparams = default_vqvae_opt_hparams.update(
    weight_decay=0.0,
)
