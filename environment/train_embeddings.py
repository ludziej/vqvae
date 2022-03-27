from torch.utils.data import DataLoader
from pathlib import Path
from environment.dataset import MusicDataset
from vqvae.model import VQVAE
import logging
from utils.misc import lazy_compute_pickle
from old_ml_utils.audio_utils import calculate_bandwidth
from environment.train_utils import generic_train, get_last_path, set_logger


def create_vqvae(l_mu, from_last_checkpot, ckpt_dir, restore_ckpt, main_dir, **params):
    all_params = dict(input_channels=1, mu=l_mu, **params)
    last_path = get_last_path(main_dir, ckpt_dir, restore_ckpt) if from_last_checkpot else None
    logging.info(f"Restoring VQVAE from {last_path}" if last_path else f"Starting VQVAE training from scratch")
    model = VQVAE.load_from_checkpoint(last_path, **all_params) if last_path is not None else VQVAE(**all_params)
    return model


def calc_dataset_dependent_params(dataset, forward_params, duration):
    forward_params.bandwidth = lazy_compute_pickle(
        lambda: calculate_bandwidth(dataset, duration=duration, hps=forward_params),
        dataset.cache_dir / forward_params.bandwidth_cache)


def get_model(sample_len, sr, train_path, forward_params, band_est_dur, use_audiofile, chunk_timeout,
              chunk_another_thread, with_train_data=False, **params):
    train_data = MusicDataset(train_path, sample_len=sample_len, sr=sr, use_audiofile=use_audiofile,
                              timeout=chunk_timeout, another_thread=chunk_another_thread)
    logging.info(train_data)
    calc_dataset_dependent_params(train_data, forward_params, band_est_dur)
    params["forward_params"] = forward_params
    params["sr"] = sr
    model = create_vqvae(**params)
    return (model, train_data) if with_train_data else model


def get_model_with_data(batch_size, sample_len, num_workers, sr, shuffle_data, test_path=None, **params):
    model, train_data = get_model(sample_len, sr, with_train_data=True, **params)
    if test_path is not None:
        test_data = MusicDataset(test_path, sample_len=sample_len, sr=sr)
    else:
        train_data, test_data = train_data.split_into_two(test_perc=0.1)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle_data)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle_data)
    return model, train_dataloader, test_dataloader


def train(hparams):
    root_dir = Path(hparams.vqvae.main_dir)
    set_logger(root_dir, hparams)
    model, train_dataloader, test_dataloader =\
        get_model_with_data(**hparams.vqvae, train_path=hparams.train_path, test_path=hparams.test_path)
    generic_train(model, hparams, train_dataloader, test_dataloader, hparams.vqvae, root_dir)
