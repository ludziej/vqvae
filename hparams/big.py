from hparams.parser import Hparams
from hparams.small import forward_params, vqvae_opt_hparams, dirs_config, small_transformer_params

big_vqvae_model_params = Hparams(
    levels=3,
    loss_fn="lmix",
    downs_t=(3, 2, 2),
    strides_t=(2, 2, 2),
    emb_width=64,
    l_bins=2048,
    l_mu=0.99,
    commit=0.02,
    spectral=0.0,
    multispectral=1.0,
    multipliers=(2, 1, 1),
    width=32,
    depth=4,
    m_conv=1.0,
    dilation_growth_rate=3,
    batch_size=4,
    sample_len=1048576,
    num_workers=5,
    sr=44100,
    group_norm=False,
    norm_in_wavenet=False,
    forward_params=forward_params,
    from_last_checkpot=True,
    main_dir="generated/models/big_vqvae/",
)

big_vqvae_params = Hparams(**vqvae_opt_hparams.__dict__, **big_vqvae_model_params.__dict__)

big_transformer_params = Hparams(
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

big_prior_params = Hparams(
    **big_transformer_params.__dict__,
    n_ctx=1540,
    main_dir="generated/prior/",
    context_on_level=False,
    log_starting_context_len=390,
    res_scale=False,
)

big_upsampler_conditioner_params = Hparams(
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

big_upsampler_params = Hparams(
    **small_transformer_params.__dict__,
    n_ctx=1560,
    main_dir="generated/upsampler/",
    context_on_level=True,
    log_context_time=49920,  # 2.5 s
    conds_kwargs=big_upsampler_conditioner_params,
)