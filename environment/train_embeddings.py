from torch.utils.data import DataLoader
from environment.dataloaders import WaveDataset
from vqvae.model import VQVAE
from old_ml_utils.audio_utils import calculate_bandwidth
from environment.train_utils import generic_train, get_last_path


def create_vqvae(sample_length, l_mu, from_last_checkpot, ckpt_dir, restore_ckpt, main_dir, **params):
    all_params = {"input_shape": (sample_length, 1), "mu": l_mu, **params}
    last_path = get_last_path(main_dir, ckpt_dir, restore_ckpt) if from_last_checkpot else None
    print(f"Restoring VQVAE from {last_path}" if last_path else f"Starting VQVAE training from scratch")
    model = VQVAE.load_from_checkpoint(last_path, **all_params) if last_path is not None else VQVAE(**all_params)
    return model


def calc_dataset_dependent_params(dataset, forward_params, duration):
    forward_params.bandwidth = calculate_bandwidth(dataset, duration=duration, hps=forward_params)


def get_model(sample_len, data_depth, sr, train_path, forward_params, band_est_dur, with_train_data=False, **params):
    train_data = WaveDataset(train_path, sample_len=sample_len, depth=data_depth, sr=sr)
    print(train_data)
    calc_dataset_dependent_params(train_data, forward_params, band_est_dur)
    params["forward_params"] = forward_params
    params["sr"] = sr
    del params["sample_length"]
    model = create_vqvae(sample_len, **params)
    return (model, train_data) if with_train_data else model


def get_model_with_data(batch_size, sample_len, num_workers,  data_depth, sr, test_path=None, **params):
    model, train_data = get_model(sample_len, data_depth, sr, with_train_data=True, **params)
    if test_path is not None:
        test_data = WaveDataset(test_path, sample_len=sample_len, depth=data_depth, sr=sr)
    else:
        train_data, test_data = train_data.split_into_two(test_perc=0.1)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers//2, shuffle=False)
    return model, train_dataloader, test_dataloader


def train(hparams):
    model, train_dataloader, test_dataloader =\
        get_model_with_data(**hparams.vqvae, train_path=hparams.train_path,
                            data_depth=hparams.data_depth, test_path=hparams.test_path)

    generic_train(model, hparams, train_dataloader, test_dataloader, hparams.vqvae)
