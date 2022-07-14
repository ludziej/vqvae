from hparams.defaults import *


big_forward_params = default_forward_params.update(
    lmix_l1=0.02,
    lmix_l2=1.0,
    lmix_linf=0,
)

big_vqvae_model_params = default_vqvae_model_params.update(
    levels=3,
    loss_fn="lmix",
    downs_t=(3, 2, 2),
    strides_t=(2, 2, 2),
    emb_width=64,
    l_bins=2048,
    commit=0.02,
    spectral=0.0,
    multispectral=1.0,
    multipliers=(2, 1, 1),
    width=32,
    depth=4,
    dilation_growth_rate=3,
    batch_size=4,
    sample_len=1048576,
    sr=44100,
    num_workers=5,
    norm_type="none",
    with_discriminator=False,
    forward_params=big_forward_params,
    main_dir="generated/models/big_vqvae/",
)

big_vqvae_opt_hparams = default_vqvae_opt_hparams.update(
    epochs=10000,
    lr=0.0003,
    weight_decay=1e-4,
    lr_warmup=100.0,
)


big_vqvae_params = Hparams(**big_vqvae_opt_hparams.__dict__, **big_vqvae_model_params.__dict__)

big_transformer_params = default_transformer_params.update(
    dim=512,
    depth=4,
    heads=4,
    dim_head=64,
    lr=0.0003,
    log_sample_size=(2, 770),  # 10 s, for prior only
    init_bins_from_vqvae=False,
    layer_for_logits=True,
    norm_type="none",
    sch_factor=0.9,
    conditioning_dropout=0.,
    log_interval=5000,
    conds_kwargs=None,
    log_context_time=0,  # 2.5 s
)

big_upsampler_conditioner_params = default_upsampler_conditioner_params.update(
    depth=16,
    width=1024,
    dilation_growth_rate=3,
    norm_type="none",
)

big_prior_params = big_transformer_params.update(
    n_ctx=1540,
    main_dir="generated/big_prior/",
    context_on_level=False,
    log_starting_context_len=390,
    res_scale=False,
)

big_upsampler_params = big_transformer_params.update(
    n_ctx=1560,
    main_dir="generated/big_upsampler/",
    context_on_level=True,
    conds_kwargs=big_upsampler_conditioner_params,
    log_context_time=49920,  # 2.5 s
)