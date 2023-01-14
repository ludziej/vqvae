from hparams.parser import Hparams
from hparams.generator.transformer import big_transformer_params, small_transformer_params
from hparams.generator.prior import big_prior_params, small_prior_params
from hparams.compressor.model import big_vqvae_params, small_vqvae_params
from hparams.compressor.forward import small_forward_params
from hparams.compressor.optim import small_vqvae_opt_hparams
from hparams.generator.upsampler import small_upsampler_conditioner_params, small_upsampler_params, \
    big_upsampler_conditioner_params, big_upsampler_params
from hparams.compressor.vae import big_vae_params, small_vae_params
from hparams.compressor.misc import dirs_config
from hparams.compressor.diffusion import big_diffusion_params


config_big_hparams = Hparams(
    model="compressor",
    level=0,
    upsampler=[big_upsampler_params, big_upsampler_params],
    prior=big_prior_params,
    compressor=big_vqvae_params,
    diffusion=big_diffusion_params,
    gpus=[0],
    train_path="resources/string_quartets/preprocessed",
    test_path=None,
    accelerator='ddp',
    max_epochs=50000,
    max_steps=0,
    gradient_clip_val=5,
    logging="INFO",
    log_file="logs.txt",
    track_grad_norm=-1,
    log_every_n_steps=10,
    restore_training=True,
    detect_anomaly=True,
    device_stats=False,
    fp16=False,
)

config_small_hparams = Hparams(
    model="compressor",
    level=0,
    upsampler=[small_upsampler_params, small_upsampler_params],
    prior=small_prior_params,
    compressor=small_vqvae_params,
    accelerator='ddp',
    gpus=[0],
    train_path="resources/music_data/",
    test_path=None,
    max_epochs=50000,
    max_steps=0,
    gradient_clip_val=5,
    logging="INFO",
    log_file="logs.txt",
    track_grad_norm=-1,
    log_every_n_steps=10,
    restore_training=True,
    detect_anomaly=True,
    device_stats=False,
    fp16=False,
)

config_gen_big_hparams = Hparams(
    upsampler=[big_upsampler_params, big_upsampler_params],
    prior=big_prior_params,
    compressor=big_vqvae_params,
    gpus=[0],
    logging="INFO",
    logger_dir="generated/gen_logs/",
    train_path="resources/",  # needed to estimate stats
    log_file="logs.txt",
    action="encdec",  # ["encdec", "continue", "generate"]
    filepaths=("in_file.wav",),
    out_path="out",
    level=0,
    bs=1,
    sec_from=-1,
    with_tqdm=True,
    artist=-1,  # -1 means randomize
    time=-1,  # seconds
    bpm=-1,
)

hparams_registry = dict(
    gen_big=config_gen_big_hparams,
    default=config_small_hparams,
    config_big=config_big_hparams,
    config_small=config_small_hparams,
    dirs_config=dirs_config,
    small_forward=small_forward_params,
    small_vqvae_opt=small_vqvae_opt_hparams,
    small_vqvae=small_vqvae_params,
    big_vqvae=big_vqvae_params,
    big_vae=big_vae_params,
    small_vae=small_vae_params,
    big_diffusion=big_diffusion_params,

    transformer=small_transformer_params,
    big_transformer=big_transformer_params,
    big_upsampler_conditioner=big_upsampler_conditioner_params,
    small_upsampler_conditioner=small_upsampler_conditioner_params,
    small_upsampler=small_upsampler_params,
    big_upsampler=big_upsampler_params,
    small_prior=small_prior_params,
    big_prior=big_prior_params,
)