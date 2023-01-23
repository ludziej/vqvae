from hparams.parser import Hparams
from hparams.compressor.misc import dirs_config


default_diffusion_autenc_params = Hparams(
    levels=1,
    downs_t=(3,),
    strides_t=(2,),
    emb_width=64,
    multipliers=(1,),
    width=64,
    depth=8,
    m_conv=1.0,
    dilation_growth_rate=3,
    norm_type="none",
    use_weight_standard=False,
    log_weights_norm=True,
    skip_valid_logs=True,
    from_last_checkpot=True,
    skip_connections=True,
    mu=0.99,
    norm_before_vqvae=False,
    l_bins=2000,
    leaky_param=1e-2,
    num_groups=32,
    bottleneck_type="none",  # ["none", "vqvae", "vae"]
)

default_diffusion_train_params = Hparams(
    noise_steps=1000,
    beta_start=1e-4,
    beta_end=0.02,
    renormalize_sampling=False,
)

default_diffusion_optim_params = Hparams(
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
)

default_diffusion_params = Hparams(
    autenc_params=default_diffusion_autenc_params,
    diff_params=default_diffusion_train_params,
    opt_params=default_diffusion_optim_params,
    log_sample_bs=5,
    log_interval=100,
    max_logged_sounds=5,
    prep_chunks=2,
    prep_level=0,
    n_ctx=4048,
    bottleneck_t_weight=0.1,
    main_dir="generated/models/big_diffusion/",
    ckpt_freq=10,
    **dirs_config,
)

big_diffusion_params = default_diffusion_params.update(

)