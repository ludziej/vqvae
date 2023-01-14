from environment.train_embeddings import get_model_with_data
from generator.model import LevelGenerator
from environment.train_utils import generic_train, create_logger, generic_get_model
from vqvae.modules.helpers import get_sample_len_from_tokens


def get_model(**params):
    return generic_get_model("performer", LevelGenerator, **params)


def train_generator(hparams, model_params, level):
    logger, root_dir = create_logger(model_params.main_dir, hparams, level=level)

    # calculate correct sample_len for chosen n_ctx inside tokens
    hparams.compressor.sample_len = get_sample_len_from_tokens(hparams.compressor.strides_t, hparams.compressor.downs_t,
                                                               level, model_params.n_ctx)
    compressor, train_dl, test_dl = get_model_with_data(**hparams.compressor, train_path=hparams.train_path,
                                                        test_path=hparams.test_path, logger=logger)
    prior = get_model(preprocessing=compressor, level=level, **model_params, logger=logger)
    generic_train(prior, hparams, train_dl, test_dl, model_params, root_dir)


def train_prior(hparams):
    train_generator(hparams, hparams.prior, hparams.compressor.levels - 1)


def train_upsampler(hparams):
    train_generator(hparams, hparams.upsampler[hparams.level], hparams.level)
