from hparams.generator.transformer import default_transformer_params, small_transformer_params, big_transformer_params
from hparams.parser import Hparams

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

small_upsampler_conditioner_params = default_upsampler_conditioner_params.update(
)
small_upsampler_params = Hparams(
    **small_transformer_params.__dict__,
    n_ctx=1560,
    main_dir="generated/small_upsampler/",
    context_on_level=True,
    log_context_time=49920,  # 2.5 s
    conds_kwargs=small_upsampler_conditioner_params,
)

big_upsampler_conditioner_params = default_upsampler_conditioner_params.update(
)
big_upsampler_params = big_transformer_params.update(
    n_ctx=1560,
    main_dir="generated/big_upsampler/",
    context_on_level=True,
    conds_kwargs=big_upsampler_conditioner_params,
    log_context_time=49920,  # 2.5 s
)
