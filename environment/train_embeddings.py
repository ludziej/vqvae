from torch.utils.data import DataLoader
from data_processing.dataset import MusicDataset
from vqvae.model import WavCompressor
from utils.misc import lazy_compute_pickle
from utils.old_ml_utils.audio_utils import calculate_bandwidth
from environment.train_utils import generic_train, create_logger, generic_get_model


def create_vqvae(l_mu, logger, **params):
    return generic_get_model("Compressor", WavCompressor, input_channels=1, mu=l_mu, logger=logger, **params)


def calc_dataset_dependent_params(dataset, forward_params, duration):
    forward_params.bandwidth = lazy_compute_pickle(
        lambda: calculate_bandwidth(dataset, duration=duration, hps=forward_params),
        dataset.cache_dir / forward_params.bandwidth_cache)
    return forward_params


def get_model(sample_len, sr, train_path, forward_params, band_est_dur, use_audiofile, chunk_timeout, logger,
              chunk_another_thread, with_train_data=False, rms_normalize_sound=True, rms_normalize_level=-5, **params):
    train_data = MusicDataset(train_path, sample_len=sample_len, sr=sr, use_audiofile=use_audiofile, logger=logger,
                              timeout=chunk_timeout, another_thread=chunk_another_thread,
                              rms_normalize_sound=rms_normalize_sound, rms_normalize_level=rms_normalize_level)
    logger.info(train_data)
    forward_params = calc_dataset_dependent_params(train_data, forward_params, band_est_dur)
    model = create_vqvae(**params, sr=sr, forward_params=forward_params,
                         rms_normalize_level=rms_normalize_level, logger=logger)
    return (model, train_data) if with_train_data else model


def get_model_with_data(batch_size, sample_len, num_workers, sr, shuffle_data, logger, test_perc, prefetch_data,
                        test_path=None, **params):
    model, train_data = get_model(sample_len, sr, with_train_data=True, logger=logger, **params)
    if test_path is not None:
        raise Exception("Not implemented")
        test_data = MusicDataset(test_path, sample_len=sample_len, sr=sr, logger=logger)
    else:
        train_data, test_data = train_data.split_into_two(test_perc=test_perc)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle_data,
                                  prefetch_factor=prefetch_data)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                 prefetch_factor=prefetch_data)
    return model, train_dataloader, test_dataloader


def train(hparams):
    logger, root_dir = create_logger(hparams.compressor.main_dir, hparams)
    model, train_dataloader, test_dataloader =\
        get_model_with_data(**hparams.compressor, train_path=hparams.train_path, test_path=hparams.test_path,
                            logger=logger)
    generic_train(model, hparams, train_dataloader, test_dataloader, hparams.compressor, root_dir)
