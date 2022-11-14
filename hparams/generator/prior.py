from hparams.generator.transformer import default_transformer_params, big_transformer_params, small_transformer_params
from hparams.parser import Hparams

default_prior_params = Hparams(
    **default_transformer_params.__dict__,
    n_ctx=1540,
    main_dir="generated/prior/",
    context_on_level=False,
    log_starting_context_len=390,
    res_scale=False,
)

big_prior_params = big_transformer_params.update(
    n_ctx=1540,
    main_dir="generated/big_prior/",
    context_on_level=False,
    log_starting_context_len=390,
    res_scale=False,
)

small_prior_params = Hparams(
    **small_transformer_params.__dict__,
    n_ctx=1540,
    main_dir="generated/small_prior/",
    context_on_level=False,
    log_starting_context_len=390,
    res_scale=False,
)
