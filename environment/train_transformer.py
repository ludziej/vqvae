from environment.train_embeddings import get_model_with_data
from generator.model import Prior
from environment.train_utils import generic_train, get_last_path


def get_model(ckpt_dir, restore_ckpt, **params):
    last_path = get_last_path(ckpt_dir, restore_ckpt)
    print(f"Restoring performer from {last_path}" if last_path else f"Starting training from scratch")
    model = Prior.load_from_checkpoint(last_path, **params) if last_path is not None else Prior(**params)
    return model


def train_prior(hparams):
    vqvae, train_dataloader, test_dataloader =\
        get_model_with_data(**hparams.vqvae, train_path=hparams.train_path,
                            data_depth=hparams.data_depth, test_path=hparams.test_path)
    prior = get_model(vqvae=vqvae, **hparams.prior)
    generic_train(prior, hparams, train_dataloader, test_dataloader, hparams.prior)


def train_upsampler(hparams, level=0):
    vqvae, train_dataloader, test_dataloader =\
        get_model_with_data(**hparams.vqvae, train_path=hparams.train_path,
                            data_depth=hparams.data_depth, test_path=hparams.test_path)
    prior = get_model(vqvae=vqvae, **hparams.upsampler[level])
    generic_train(prior, hparams, train_dataloader, test_dataloader, hparams.upsampler[level])
