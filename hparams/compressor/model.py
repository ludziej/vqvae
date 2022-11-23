from hparams.compressor.forward import default_forward_params, big_forward_params, small_forward_params
from hparams.parser import Hparams
from hparams.compressor.adversarial import adv_params
from hparams.compressor.optim import default_vqvae_opt_hparams, big_vqvae_opt_hparams, small_vqvae_opt_hparams

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
default_vqvae_params = Hparams(**default_vqvae_opt_hparams.__dict__, **default_vqvae_model_params.__dict__)

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
    forward_params=big_forward_params,
    main_dir="generated/models/big_vqvae/",
)
big_vqvae_params = Hparams(**big_vqvae_opt_hparams.__dict__, **big_vqvae_model_params.__dict__)

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
    norm_type="none",
    forward_params=small_forward_params,
    from_last_checkpot=True,
    main_dir="generated/models/small_vqvae/",
)
small_vqvae_params = Hparams(**small_vqvae_opt_hparams.__dict__, **small_vqvae_model_params.__dict__)
