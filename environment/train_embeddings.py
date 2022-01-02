import numpy as np
from torch.utils.data import DataLoader
from environment.dataloaders import WaveDataset
from vqvae.model import VQVAE
from pytorch_lightning import Trainer
from old_ml_utils.audio_utils import calculate_bandwidth
from itertools import takewhile, count
from functools import reduce
import torch
import os
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint


def get_last_path(dir="lightning_logs", version="version_", checkpoint_name=""):
    #("{}/{}{}/".format(dir, version, i) for i in count())
    #files = takewhile(os.path.exists, )
    #last = reduce(lambda x, y: y, files, None)
    file = "generated/best_checkpoint/best_model.ckpt"
    return file if os.path.exists(file) else None


def create_vqvae(sample_length, l_mu, from_last_checkpot, **params):
    all_params = {"input_shape": (sample_length, 1), "mu": l_mu, **params}
    last_path = get_last_path() if from_last_checkpot else None
    print(f"Restoring model from {last_path}" if last_path else f"Starting training from scratch")
    model = VQVAE.load_from_checkpoint(last_path, **all_params) if last_path is not None else VQVAE(**all_params)
    return model


def calc_metaparams(dataset, forward_params, duration=1000):
    forward_params.bandwidth = calculate_bandwidth(dataset, duration=duration, hps=forward_params)


def get_model(sample_len, data_depth, sr, train_path, forward_params, with_train_data=False, **params):
    train_data = WaveDataset(train_path, sample_len=sample_len, depth=data_depth, sr=sr)
    print(train_data)
    calc_metaparams(train_data, forward_params)
    params["forward_params"] = forward_params
    params["sr"] = sr
    model = create_vqvae(sample_len, **params)
    return (model, train_data) if with_train_data else model


def train(batch_size, sample_len, num_workers,  data_depth, sr, gpus, test_path=None, **params):
    model, train_data = get_model(sample_len, data_depth, sr, with_train_data=True, **params)
    if test_path is not None:
        test_data = WaveDataset(test_path, sample_len=sample_len, depth=data_depth, sr=sr)
    else:
        train_data, test_data = train_data.split_into_two(test_perc=0.1)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers//2, shuffle=True)

    checkpoint_callback = ModelCheckpoint(dirpath='generated/best_checkpoint/', every_n_epochs=10,
                                          filename='best_model-{epoch}-{val_multispectral_loss_epoch:.2f}-{spectral_loss_epoch:.2f}')
    tb_logger = pl_loggers.TensorBoardLogger("generated/logs/")
    trainer = Trainer(gpus=gpus, log_every_n_steps=1, logger=tb_logger, default_root_dir="generated/checkpoints", accelerator='dp',
                      callbacks=[checkpoint_callback], max_epochs=50000)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
