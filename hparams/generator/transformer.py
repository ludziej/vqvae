from hparams.compressor.misc import dirs_config
from hparams.parser import Hparams


opt_params = Hparams(
    lr_warmup=100.0,
    lr_decay=3000,
    lr_gamma=0.3,
    lr_scale=1.0,
    lr_use_linear_decay=False,
    lr_start_linear_decay=0,
)

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
    pos_enc_type="fourier",  # [trainable, fourier, bpm]
    scheduler_type="step",  # ["plateau", "step", "none"]
    pos_enc_lvl_over_bit=4,
    sch_patience=1000,
    sch_factor=0.9,
    conditioning_dropout=0.,
    log_interval=5000,
    #ckpt_name="model-{epoch}-{val_loss:.2f}-{loss:.2f}",
    ckpt_name='last_model',
    conditioning_concat=False,
    opt_params=opt_params,
    **dirs_config.__dict__,
)

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
    log_context_time=0,  # 2.5 s
)

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
    sch_factor=0.9,
    conditioning_dropout=0.,
    log_interval=5000,
    ckpt_name="model-{epoch}-{val_loss:.2f}-{loss:.2f}",
    **dirs_config.__dict__,
)