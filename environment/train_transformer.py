from environment.train_embeddings import get_model_with_data
from generator.model import LevelGenerator
from environment.train_utils import generic_train, get_last_path


def get_model(main_dir, ckpt_dir, restore_ckpt, **params):
    last_path = get_last_path(main_dir, ckpt_dir, restore_ckpt)
    print(f"Restoring performer from {last_path}" if last_path else f"Starting performer training from scratch")
    model = LevelGenerator.load_from_checkpoint(last_path, **params) if last_path is not None else LevelGenerator(**params)
    return model


def train_generator(hparams, model_params):
    vqvae, train_dataloader, test_dataloader = \
        get_model_with_data(**hparams.vqvae, train_path=hparams.train_path,
                            data_depth=hparams.data_depth, test_path=hparams.test_path)
    prior = get_model(vqvae=vqvae, **model_params)
    generic_train(prior, hparams, train_dataloader, test_dataloader, model_params)


def train_prior(hparams):
    model_params = hparams.prior
    hparams.vqvae.sample_length = model_params.sample_len
    train_generator(hparams, model_params)


def train_upsampler(hparams, level=0):
    model_params = hparams.upsampler[level]
    hparams.vqvae.sample_length = model_params.sample_len
    train_generator(hparams, model_params)
