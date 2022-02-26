from types import SimpleNamespace
from collections.abc import Mapping


class Hparams(SimpleNamespace, Mapping):
    def iter(self, fun, modify=False):
        def monad(value, key=None):
            if isinstance(value, Hparams):  # parse substructure
                for new_key, new_val in value.items():
                    x = monad(new_val, f"{key}.{new_key}" if key is not None else new_key)
                    if modify:
                        value[new_key] = x
                return value
            return fun(key, value)
        return monad(self)

    def items(self):
        return self.__dict__.items()

    def __iter__(self):
        return self.__dict__.__iter__()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        return self.__dict__.__setitem__(key, value)


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
    bandwidth={}
)

smallvqvae_params = Hparams(
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
    width=32,
    depth=4,
    m_conv=1.0,
    dilation_growth_rate=3,
    batch_size=4,
    sample_len=262144,
    num_workers=5,
    sr=22050,
    forward_params=forward_params,
    from_last_checkpot=True,
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
    main_dir="generated/vqvae/",
    ckpt_name='model-{epoch}-{val_multispectral_loss_epoch:.2f}-{spectral_loss_epoch:.2f}',
    **dirs_config.__dict__,
)

vqvae_params = Hparams(**vqvae_opt_hparams.__dict__, **smallvqvae_params.__dict__)

transformer_params = Hparams(
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
    groupnorm=False,
    ckpt_name="model-{epoch}-{val_loss:.2f}-{loss:.2f}",
    **dirs_config.__dict__,
)


smallprior_params = Hparams(
    **transformer_params.__dict__,
    n_ctx=1540,
    sample_len=49920,
    main_dir="generated/prior/",
    level=1,
    context_on_level=False,
    log_starting_context_len=390,
    res_scale=False,
)

upsampler_conditioner_params = Hparams(
    res_scale=True,
    depth=16,
    width=1024,
    init_scale=1,
    m_conv=1.0,
    dilation_growth_rate=3,
    dilation_cycle=8,
    checkpoint_res=0,
    zero_out=False,
)

smallupsampler_params = Hparams(
    **transformer_params.__dict__,
    n_ctx=1560,
    sample_len=262144,
    main_dir="generated/upsampler/",
    level=0,
    context_on_level=True,
    log_context_time=49920,  # 2.5 s
    conds_kwargs=upsampler_conditioner_params,
)

default_hparams = Hparams(
    model="vqvae",
    upsampler=[smallupsampler_params],
    prior=smallprior_params,
    vqvae=vqvae_params,
    gpus=[0],
    train_path="resources/string_quartets/preprocessed",
    test_path=None,
    data_depth=1,
    accelerator='dp',
    max_epochs=50000,
)







