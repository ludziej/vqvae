import os
import logging
import sys
from datetime import datetime
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor, DeviceStatsMonitor
from pathlib import Path
import json
import sys


class NoSmoothingTQDM(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.smoothing = 0
        return bar


def generic_train(model, hparams, train, test, model_hparams, root_dir):
    tb_logger = pl_loggers.TensorBoardLogger(root_dir / model_hparams.logs_dir)
    checkpoint_callback = ModelCheckpoint(dirpath=root_dir / model_hparams.ckpt_dir, filename=model_hparams.ckpt_name,
                                          every_n_train_steps=model_hparams.ckpt_freq)
    tqdm_pb = NoSmoothingTQDM()
    lr_monitor = LearningRateMonitor(logging_interval='step')
    device_stats = DeviceStatsMonitor()
    callbacks = [checkpoint_callback, tqdm_pb, lr_monitor, device_stats]

    trainer = Trainer(gpus=hparams.gpus, profiler="simple", max_epochs=hparams.max_epochs,
                      max_steps=hparams.max_steps if hparams.max_steps != 0 else -1,
                      gradient_clip_val=hparams.gradient_clip_val, callbacks=callbacks,
                      log_every_n_steps=1, logger=tb_logger, strategy=hparams.accelerator, detect_anomaly=True,
                      default_root_dir=root_dir / model_hparams.default_ckpt_root, track_grad_norm=2)
    trainer.fit(model, train_dataloaders=train, val_dataloaders=test)


def get_last_path(main_dir, ckpt_dir, best_ckpt):
    file = Path(main_dir) / ckpt_dir / best_ckpt
    return file if os.path.exists(file) else None


def save_hparams(root_path, hparams, filename):
    whole_save = dict(cli_args=sys.argv, hparams=hparams)
    with open(root_path / filename, 'w') as f:
        json.dump(whole_save, fp=f, default=lambda obj: obj.__dict__, indent=4)


def create_logger(root_dir, hparams, hparams_file="hparams.json"):
    root_dir.mkdir(parents=True, exist_ok=True)
    strtime = datetime.now().strftime("%d.%m.%Y, %H:%M:%S")
    save_hparams(root_dir, hparams, f"{strtime} {hparams_file}")
    logging.basicConfig(format="%(asctime)s [%(threadName)-1s] [%(thread)-1s] [%(levelname)-1s]  %(message)s")
#    logging.basicConfig(level=hparams.logging,
#                        format="%(asctime)s [%(threadName)-1s] [%(thread)-1s] [%(levelname)-1s]  %(message)s",
#                        handlers=[logging.StreamHandler(sys.stdout),
#                                  logging.FileHandler(root_dir / f"{strtime} {hparams.log_file}")],)
    logger = logging.getLogger(__name__)
    logger.setLevel(hparams.logging)
    logger.addHandler(logging.FileHandler(root_dir / f"{strtime} {hparams.log_file}"))
    return logger
