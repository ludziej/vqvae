import os
import logging
import sys
from datetime import datetime
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path


def generic_train(model, hparams, train, test, model_hparams, root_dir):
    tb_logger = pl_loggers.TensorBoardLogger(root_dir / model_hparams.logs_dir)
    checkpoint_callback = ModelCheckpoint(dirpath=root_dir / model_hparams.ckpt_dir, filename=model_hparams.ckpt_name,
                                          every_n_epochs=model_hparams.ckpt_freq)

    trainer = Trainer(gpus=hparams.gpus, profiler="simple", max_epochs=hparams.max_epochs,
                      max_steps=hparams.max_steps if hparams.max_steps != 0 else -1,
                      log_every_n_steps=1, logger=tb_logger, strategy=hparams.accelerator,
                      default_root_dir=root_dir / model_hparams.default_ckpt_root, callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloaders=train, val_dataloaders=test)


def get_last_path(main_dir, ckpt_dir, best_ckpt):
    file = Path(main_dir) / ckpt_dir / best_ckpt
    return file if os.path.exists(file) else None


def set_logger(root_dir, hparams):
    root_dir.mkdir(parents=True, exist_ok=True)
    strtime = datetime.now().strftime("%d.%m.%Y, %H:%M:%S")
    logging.basicConfig(level=hparams.logging,
                        format="%(asctime)s [%(threadName)-1s] [%(levelname)-1s]  %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(root_dir / f"{strtime} {hparams.log_file}")],)
