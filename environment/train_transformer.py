from pathlib import Path
import logging

from environment.train_embeddings import get_model_with_data
from generator.model import LevelGenerator
from environment.train_utils import generic_train, get_last_path, set_logger
from vqvae.helpers import get_sample_len_from_tokens


def get_model(main_dir, ckpt_dir, restore_ckpt, **params):
    last_path = get_last_path(main_dir, ckpt_dir, restore_ckpt)
    logging.info(f"Restoring performer from {last_path}" if last_path else f"Starting performer training from scratch")
    model = LevelGenerator.load_from_checkpoint(last_path, **params) \
        if last_path is not None else LevelGenerator(**params)
    return model


def train_generator(hparams, model_params, level):
    root_dir = Path(model_params.main_dir)
    root_dir = root_dir / str(level) if hparams.model == "upsampler" else root_dir
    root_dir.mkdir(parents=True, exist_ok=True)
    set_logger(root_dir, hparams)

    # calculate correct sample_len for chosen n_ctx inside tokens
    hparams.vqvae.sample_len = get_sample_len_from_tokens(hparams.vqvae.strides_t, hparams.vqvae.downs_t,
                                                          level, model_params.n_ctx)
    vqvae, train_dataloader, test_dataloader = \
        get_model_with_data(**hparams.vqvae, train_path=hparams.train_path,
                            data_depth=hparams.data_depth, test_path=hparams.test_path)
    prior = get_model(vqvae=vqvae, level=level, **model_params)
    generic_train(prior, hparams, train_dataloader, test_dataloader, model_params, root_dir)


def train_prior(hparams):
    train_generator(hparams, hparams.prior, hparams.vqvae.levels - 1)


def train_upsampler(hparams):
    train_generator(hparams, hparams.upsampler[hparams.level], hparams.level)
