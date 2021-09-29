import numpy as np
from torch.utils.data import DataLoader
from environment.dataloaders import WaveDataset
from model.vqvae import VQVAE
from pytorch_lightning import Trainer
from torch.utils.data import random_split
from ml_utils.audio_utils import calculate_bandwidth


def create_vqvae(sample_length, l_mu, **params):
    return VQVAE(
        input_shape=(sample_length, 1),
        mu=l_mu,
        **params)


def calc_metaparams(dataset, forward_params, duration=30):
    forward_params.bandwidth = calculate_bandwidth(dataset[0].numpy().T, duration=duration, hps=forward_params)


def train(batch_size, sample_len, num_workers, train_path, data_depth, sr, gpus, forward_params, test_path=None, **params):
    train_data = WaveDataset(train_path, sample_len=sample_len, depth=data_depth, sr=sr)
    calc_metaparams(train_data, forward_params)
    if test_path is not None:
        test_data = WaveDataset(test_path, sample_len=sample_len, depth=data_depth, sr=sr)
    else:
        train_size = int(0.9 * len(train_data))
        train_data, test_data = random_split(train_data, [train_size, len(train_data) - train_size])
    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
    params["forward_params"] = forward_params
    model = create_vqvae(sample_len, **params)

    trainer = Trainer(gpus=gpus, log_every_n_steps=50)
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=test_dataloader)
