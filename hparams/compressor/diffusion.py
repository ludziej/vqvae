from hparams.parser import Hparams
from hparams.compressor.misc import dirs_config
from utils.misc import flatten


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
    downs_t=(4,),
    strides_t=(2,),
    emb_width=2048,
    multipliers=(1,),
    width=128,
    depth=8,
    m_conv=1.0,
    dilation_growth_rate=2,
    norm_type="group",
    use_weight_standard=True,
    log_weights_norm=1,
    skip_valid_logs=True,
    from_last_checkpot=True,
    use_log_grads=1,
    skip_connections=True,
    skip_connections_step=1,
    channel_increase=2,
    mu=0.99,
    res_scale=1.,  # 1/(2**0.5),
    dilation_cycle=1,
    norm_before_vqvae=False,
    l_bins=2000,
    use_bias=False,
    rezero=True,
    rezero_in_attn=True,
    biggan_skip=True,
    concat_skip=True,
    leaky_param=0.01,
    swish_act=True,
    num_groups=16,
    bottleneck_type="none",  # ["none", "vqvae", "vae", "transformer"]
    self_attn_from=3,
    bottleneck_params=default_bottleneck_transformer_params,
)

default_diffusion_train_params = Hparams(
    noise_steps=1000,
    noise_schedule="cosine",
    noise_schedule_s=0.008,
    beta_start=1e-4,
    beta_end=0.02,
    clip_val=3.,
    clip_pred=True,
    clip_input=False,
    dynamic_clipping=True,
    dclip_perc=0.99,
    use_one_step=False,
    renormalize_sampling=False,
)

default_diffusion_optim_params = Hparams(
    with_ema=True,
    ema_decay=0.9999,
    lr=0.00001,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0,
    eps=1e-08,
    lr_warmup=1.0,
    lr_decay=10000000,
    lr_gamma=0.7,
    lr_scale=1.0,
    lr_use_linear_decay=False,
    lr_start_linear_decay=0,
)

default_condition_params = Hparams(
    listens_logarithm=True,
    cls_free_guidance=True,
    drop_guidance_prob=0.1,
    cfg_guid_weight=1,
    t_cond_size=128,
    pos_cond_size=64,  # 0 if without conditioning
    style_cond_size=256,
    time_cond_size=0,
    listens_cond_size=0,
    artists_cond_size=0,
    t_enc_type="fourier",  # [fourier, trainable]
    pos_enc_type="fourier",
    time_enc_type="fourier",
    listens_enc_type="fourier",
    genre_names=None,  # filled in during dataset creation
)

noise_log_intervals = flatten([
    [(1, 1000)],
    [(i, i) for i in range(1, 5)],
    [(1000 - i, 1000 - i) for i in range(5)],
    [(i*100 + 1, (i+1)*100) for i in range(10)],
    #[(1000-(i+1)*10, 1000-i*10) for i in range(10)],
    #[(i*10 + 1, (i+1)*10) for i in range(10)],
])

default_diffusion_params = Hparams(
    model_type="diffusion",
    autenc_params=default_diffusion_autenc_params,
    diff_params=default_diffusion_train_params,
    opt_params=default_diffusion_optim_params,
    condition_params=default_condition_params,
    log_sample_bs=2,
    max_logged_sounds=2,
    log_interval=1000,
    data_time_cond=True,
    data_context_cond=True,
    prep_chunks=2,
    prep_level=1,
    n_ctx=10240,
    rmse_loss_weight=0,
    eps_loss_weight=1,
    attn_pos_enc_type="fourier",  # [trainable, fourier]
    t_pos_enc="fourier",
    main_dir="generated/models/big_diffusion/",
    logger_type="neptune",
    neptune_run_id="DIF-16",
    neptune_path="training/model/checkpoints/last",
    neptune_project="diffusion",
    stats_momentum=0.9,
    renormalize_loss=False,
    no_stochastic_prep=True,
    sample_cfgw=1.5,
    ckpt_freq=1000,
    log_intervals=noise_log_intervals,
    **dirs_config,
)

big_diffusion_params = default_diffusion_params.update(

)
