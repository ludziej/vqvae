from hparams.compressor.forward import big_forward_params
from hparams.parser import Hparams
from hparams.compressor.optim import big_vqvae_opt_hparams, small_vqvae_opt_hparams
from hparams.compressor.model import big_vqvae_model_params, small_vqvae_model_params

big_vae_model_params = big_vqvae_model_params.update(
    bottleneck_type="vae",
    bottleneck_lw=1,
    main_dir="generated/models/big_vae/",
    log_vae_no_stochastic=True,
)

big_vae_opt_hparams = big_vqvae_opt_hparams.update(
)

big_vae_params = Hparams(**big_vae_opt_hparams.__dict__, **big_vae_model_params.__dict__)


small_vae_model_params = small_vqvae_model_params.update(
    bottleneck_type="vae",
    bottleneck_lw=1,
    main_dir="generated/models/small_vae/",
    log_vae_no_stochastic=True,
)

small_vae_opt_hparams = small_vqvae_opt_hparams.update(
)

small_vae_params = Hparams(**small_vae_opt_hparams.__dict__, **small_vae_model_params.__dict__)