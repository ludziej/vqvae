from hparams.parser import Hparams


default_diffusion_autenc_params = Hparams(
    levels=1,
    downs_t=(5, 3),
    strides_t=(2, 2),
    emb_width=64,
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
    norm_type="none",
    use_weight_standard=False,
    log_interval=100,
    forward_params=default_forward_params,
    skip_valid_logs=True,
    from_last_checkpot=True,
    leaky_param=1e-2,
    bottleneck_type="none",  # ["none", "vqvae", "vae"]
    main_dir="generated/models/big_diffusion/",
)

diffusion_train_params = Hparams(
    noise_steps=1000,
    beta_start=1e-4,
    beta_end=0.02,
    img_size=256,
)

default_diffusion_autenc_params = Hparams(
    autenc_params=default_diffusion_autenc_params,
    diff_params=diffusion_train_params,
    log_sample_bs=5,
    encode_chunks=3,
    prep_level=0,
)