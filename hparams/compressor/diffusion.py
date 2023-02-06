from hparams.parser import Hparams
from hparams.compressor.misc import dirs_config


default_bottleneck_transformer_params = Hparams(
    depth=8,
    heads=4,
    ff_mul=2,
    dropout=0.1,
    prenorm=False,
    rezero=True,
)

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
    use_log_grads=True,
    skip_connections=True,
    skip_connections_step=3,
    channel_increase=2,
    mu=0.99,
    dilation_cycle=9,
    norm_before_vqvae=False,
    l_bins=2000,
    use_bias=False,
    rezero=True,
    concat_skip=True,
    leaky_param=0,
    num_groups=32,
    bottleneck_type="transformer",  # ["none", "vqvae", "vae", "transformer"]
    self_attn_from=3,
    condition_size=128,
    bottleneck_params=default_bottleneck_transformer_params,
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
    rmse_loss_weight=0.5,
    eps_loss_weight=1,
    bottleneck_t_weight=1,  # None means trainable
    pos_enc_weight=1,  # None means trainable
    attn_pos_enc_type="fourier",  # [trainable, fourier]
    t_pos_enc="fourier",
    main_dir="generated/models/big_diffusion/",
    ckpt_freq=10,
    **dirs_config,
)

big_diffusion_params = default_diffusion_params.update(

)