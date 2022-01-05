import os

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint


def generic_train(model, hparams, train, test, model_type):
    checkpoint_callback = ModelCheckpoint(dirpath=hparams[model_type].ckpt_dir, filename=hparams[model_type].ckpt_name,
                                          every_n_epochs=hparams[model_type].ckpt_name)
    tb_logger = pl_loggers.TensorBoardLogger(hparams[model_type].logs_dir)
    trainer = Trainer(gpus=hparams.gpus, log_every_n_steps=1, logger=tb_logger, accelerator=hparams.accelerator,
                      max_epochs=hparams.max_epochs,
                      default_root_dir=hparams[model_type].default_ckpt_root, callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloaders=train, val_dataloaders=test)


def get_last_path(ckpt_dir, best_ckpt):
    file = f"{ckpt_dir}{best_ckpt}"
    return file if os.path.exists(file) else None
