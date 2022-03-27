from hparams.defaults import *


small_forward_params = default_forward_params.update(
    lmix_l1=0.0,
    lmix_l2=1.0,
    lmix_linf=0.02,
)

small_vqvae_model_params = default_vqvae_model_params.update(
    levels=2,
    loss_fn="l2",
    downs_t=(5, 3),
    strides_t=(2, 2),
    emb_width=64,
    l_bins=1024,
    commit=0.02,
    spectral=0.0,
    multispectral=1.0,
    multipliers=(1, 1),
    width=32,
    depth=4,
    dilation_growth_rate=3,
    batch_size=4,
    sample_len=262144,
    sr=22050,
    num_workers=5,
    norm_before_vqvae=True,
    fixed_commit=True,
    norm_type="none",
    with_discriminator=False,
    forward_params=small_forward_params,
    from_last_checkpot=True,
    use_bottleneck=True,
    main_dir="generated/models/small_vqvae/",
)

small_vqvae_opt_hparams = default_vqvae_opt_hparams.update(
    epochs=10000,
    lr=0.0003,
    weight_decay=0.0,
    lr_warmup=100.0,
)

small_vqvae_params = Hparams(**small_vqvae_opt_hparams.__dict__, **small_vqvae_model_params.__dict__)

small_transformer_params = default_transformer_params.update(
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
    norm_type="none",
    warmup_time=1000,
    sch_patience=1000,
    sch_factor=0.9,
    conditioning_dropout=0.,
    log_interval=5000,
    ckpt_name="model-{epoch}-{val_loss:.2f}-{loss:.2f}",
    **dirs_config.__dict__,
)

small_upsampler_conditioner_params = default_upsampler_conditioner_params.update(
    depth=16,
    width=1024,
    dilation_growth_rate=3,
    norm_type="none",
)

small_prior_params = Hparams(
    **small_transformer_params.__dict__,
    n_ctx=1540,
    main_dir="generated/small_prior/",
    context_on_level=False,
    log_starting_context_len=390,
    res_scale=False,
)

small_upsampler_params = Hparams(
    **small_transformer_params.__dict__,
    n_ctx=1560,
    main_dir="generated/small_upsampler/",
    context_on_level=True,
    log_context_time=49920,  # 2.5 s
    conds_kwargs=small_upsampler_conditioner_params,
)