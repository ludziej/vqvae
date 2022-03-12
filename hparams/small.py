from hparams.parser import Hparams

forward_params = Hparams(
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
    lmix_l1=True,
    lmix_l2=True,
    lmix_linf=True,
    linf_k=2048,
    bandwidth=None,
)

small_vqvae_model_params = Hparams(
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
    multipliers=(1, 1, 1),
    width=32,
    depth=4,
    m_conv=1.0,
    dilation_growth_rate=3,
    batch_size=4,
    sample_len=262144,
    num_workers=5,
    sr=22050,
    group_norm=False,
    norm_in_wavenet=False,
    forward_params=forward_params,
    from_last_checkpot=True,
    main_dir="generated/models/small_vqvae/",
)

dirs_config = Hparams(
    restore_ckpt="best_model.ckpt",
    ckpt_dir='models',
    logs_dir="logs",
    default_ckpt_root="checkpoints",
)

vqvae_opt_hparams = Hparams(
    epochs=10000,
    lr=0.0003,
    clip=1.0,
    beta1=0.9,
    beta2=0.999,
    ignore_grad_norm=0,
    weight_decay=0.0,
    eps=1e-08,
    lr_warmup=100.0,
    lr_decay=10000000000.0,
    lr_gamma=1.0,
    lr_scale=1.0,
    lr_use_linear_decay=False,
    lr_start_linear_decay=0,
    lr_use_cosine_decay=False,
    fp16=False,
    fp16_params=False,
    fp16_loss_scale=None,
    fp16_scale_window=1000.0,
    fp16_opt=False,
    ckpt_freq=10,
    band_est_dur=1000,
    ckpt_name='model-{epoch}-{val_multispectral_loss_epoch:.2f}-{spectral_loss_epoch:.2f}',
    **dirs_config.__dict__,
)

small_vqvae_params = Hparams(**vqvae_opt_hparams.__dict__, **small_vqvae_model_params.__dict__)

small_transformer_params = Hparams(
    dim=512,
    depth=4,
    heads=4,
    dim_head=64,
    ckpt_freq=10,
    lr=0.0003,
    start_gen_sample_len=5,
    pos_init_scale=1,
    bins_init_scale=1,
    log_starting_context_perc=0.1,
    log_sample_size=(2, 770),  # 10 s, for prior only
    init_bins_from_vqvae=False,
    layer_for_logits=True,
    group_norm=False,
    warmup_time=1000,
    sch_patience=1000,
    sch_factor=0.9,
    conditioning_dropout=0.,
    log_interval=5000,
    ckpt_name="model-{epoch}-{val_loss:.2f}-{loss:.2f}",
    **dirs_config.__dict__,
)

small_prior_params = Hparams(
    **small_transformer_params.__dict__,
    n_ctx=1540,
    main_dir="generated/prior/",
    context_on_level=False,
    log_starting_context_len=390,
    res_scale=False,
)

small_upsampler_conditioner_params = Hparams(
    res_scale=True,
    depth=16,
    width=1024,
    init_scale=1,
    m_conv=1.0,
    dilation_growth_rate=3,
    dilation_cycle=8,
    checkpoint_res=0,
    zero_out=False,
    group_norm=False,
)

small_upsampler_params = Hparams(
    **small_transformer_params.__dict__,
    n_ctx=1560,
    main_dir="generated/upsampler/",
    context_on_level=True,
    log_context_time=49920,  # 2.5 s
    conds_kwargs=small_upsampler_conditioner_params,
)