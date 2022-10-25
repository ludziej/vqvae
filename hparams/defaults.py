from hparams.parser import Hparams


default_forward_params = Hparams(
    n_fft=1024,
    hop_length=256,
    window_size=1024,
    sr=22050,
    channels=2,
    wav='',
    n_inps=1,
    n_hops=2,
    n_segment=1,
    n_total_segment=1,
    n_segment_each=1,
    prime_chunks=4,
    sample_length=0,
    sample_hop_length=30000,
    max_silence_pad_length=0,
    ignore_boundaries=False,
    use_nonrelative_specloss=True,
    multispec_loss_n_fft=(2048,1024,512),
    multispec_loss_hop_length=(240,120,50),
    multispec_loss_window_size=(1200,600,240),
    lmix_l1=0.0,
    lmix_l2=1.0,
    lmix_linf=0.02,
    linf_k=2048,
    bandwidth_cache="bandwidth.cache",
    bandwidth=None,
)

adv_params = Hparams(
    with_discriminator=False,
    adv_latency=2000,  # steps
    gan_loss_weight=0.1,
    disc_use_freq=1,
    disc_loss_weight=0.1,
    gan_loss_warmup=1000,
    discriminator_level=0,
    reduce_type="avg",  # ["max", "avg"]
    type="joined",  # "mel", "wav" or "joined"
    n_mels=256,
    n_fft=4096,
    hop_length=240,
    window_size=1200,
    leaky=1e-2,
    res_depth=4,
    first_channels=32,
)

default_vqvae_model_params = Hparams(
    levels=2,
    loss_fn="l2",
    downs_t=(5, 3),
    strides_t=(2, 2),
    emb_width=64,
    l_bins=1024,
    l_mu=0.99,
    commit=0.02,
    spectral=0.0,
    multispectral=1.0,
    multipliers=(1, 1),
    width=32,
    depth=4,
    m_conv=1.0,
    dilation_growth_rate=3,
    batch_size=4,
    sample_len=262144,
    num_workers=5,
    prefetch_data=5,
    sr=22050,
    test_perc=0.1,
    norm_before_vqvae=False,
    fixed_commit=False,
    norm_type="none",
    use_weight_standard=False,
    log_interval=100,
    adv_params=adv_params,
    forward_params=default_forward_params,
    skip_valid_logs=True,
    from_last_checkpot=True,
    leaky_param=1e-2,
    use_bottleneck=True,
    main_dir="generated/models/small_vqvae/",
)

dirs_config = Hparams(
    restore_ckpt="best_model.ckpt",
    ckpt_dir='models',
    logs_dir="logs",
    default_ckpt_root="checkpoints",
)

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

default_vqvae_params = Hparams(**default_vqvae_opt_hparams.__dict__, **default_vqvae_model_params.__dict__)

default_transformer_params = Hparams(
    dim=512,
    depth=4,
    heads=4,
    dim_head=64,
    ckpt_freq=10,
    lr=0.0003,
    start_gen_sample_len=5,
    pos_init_scale=1,
    bins_init_scale=1,
    token_dim=64,
    log_starting_context_perc=0.1,
    log_sample_size=(2, 770),  # 10 s, for prior only
    init_bins_from_vqvae=False,
    layer_for_logits=True,
    norm_type="none",
    warmup_time=1000,
    feature_redraw_interval=100000,
    no_scheduler=False,
    sch_patience=1000,
    sch_factor=0.9,
    conditioning_dropout=0.,
    log_interval=5000,
    #ckpt_name="model-{epoch}-{val_loss:.2f}-{loss:.2f}",
    ckpt_name='last_model',
    **dirs_config.__dict__,
)

default_prior_params = Hparams(
    **default_transformer_params.__dict__,
    n_ctx=1540,
    main_dir="generated/prior/",
    context_on_level=False,
    log_starting_context_len=390,
    res_scale=False,
)

default_upsampler_conditioner_params = Hparams(
    res_scale=True,
    depth=16,
    width=1024,
    init_scale=1,
    m_conv=1.0,
    dilation_growth_rate=3,
    dilation_cycle=8,
    zero_out=False,
    norm_type="none",
)

default_upsampler_params = Hparams(
    **default_transformer_params.__dict__,
    n_ctx=1560,
    main_dir="generated/upsampler/",
    context_on_level=True,
    log_context_time=49920,  # 2.5 s
    conds_kwargs=default_upsampler_conditioner_params,
)