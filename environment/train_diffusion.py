from environment.train_embeddings import get_model_with_data
from vqvae.diffusion_unet import DiffusionUnet
from environment.train_utils import generic_train, create_logger, generic_get_model
from vqvae.modules.helpers import get_sample_len_from_tokens


def get_model(**params):
    return generic_get_model("diffusion", DiffusionUnet, **params)


def train(hparams):
    model_params = hparams.diffusion
    logger, root_dir = create_logger(model_params.main_dir, hparams)

    # calculate correct sample_len for chosen n_ctx inside tokens
    hparams.compressor.sample_len = get_sample_len_from_tokens(hparams.compressor.strides_t, hparams.compressor.downs_t,
                                                               model_params.prep_level, model_params.n_ctx)
    compressor, train_dl, test_dl, train_dataset = get_model_with_data(
        **hparams.compressor, train_path=hparams.train_path, test_path=hparams.test_path,
        banned_genres=hparams.banned_genres, time_cond=model_params.data_time_cond,
        context_cond=model_params.data_context_cond, logger=logger, with_train_data=True)
    model_params.condition_params.genre_names = train_dataset.genre_names

    diffusion = get_model(preprocessing=compressor, **model_params, logger=logger)
    generic_train(diffusion, hparams, train_dl, test_dl, model_params, root_dir)


