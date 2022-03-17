import os
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path


def generic_train(model, hparams, train, test, model_hparams, root_dir):
    checkpoint_callback = ModelCheckpoint(dirpath=root_dir / model_hparams.ckpt_dir, filename=model_hparams.ckpt_name,
                                          every_n_epochs=model_hparams.ckpt_freq)
    tb_logger = pl_loggers.TensorBoardLogger(root_dir / model_hparams.logs_dir)
    trainer = Trainer(gpus=hparams.gpus, profiler="simple", max_steps=20,
                      log_every_n_steps=1, logger=tb_logger, strategy=hparams.accelerator,
                      max_epochs=hparams.max_epochs,
                      default_root_dir=root_dir / model_hparams.default_ckpt_root, callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloaders=train, val_dataloaders=test)


def get_last_path(main_dir, ckpt_dir, best_ckpt):
    file = Path(main_dir) / ckpt_dir / best_ckpt
    return file if os.path.exists(file) else None
