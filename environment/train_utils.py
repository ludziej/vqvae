import os
import logging
import sys
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor, DeviceStatsMonitor
from pathlib import Path
import json
import sys
import warnings


class NoSmoothingTQDM(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.smoothing = 0
        return bar


def generic_get_model(name, model_class, main_dir, ckpt_dir, restore_ckpt, **params):
    last_path = get_last_path(main_dir, ckpt_dir, restore_ckpt)
    params["logger"].info(f"Restoring {name} from {last_path}" if last_path else
                          f"Starting {name} training from scratch")
    model = model_class.load_from_checkpoint(last_path, **params, strict=False) \
        if last_path is not None else model_class(**params)
    params["logger"].debug(f"Model {name} loaded")
    return model


def generic_train(model, hparams, train, test, model_hparams, root_dir):
    tb_logger = pl_loggers.TensorBoardLogger(root_dir / model_hparams.logs_dir)
    checkpoint_callback = ModelCheckpoint(dirpath=root_dir / model_hparams.ckpt_dir, save_top_k=0,
                                          filename=model_hparams.distinct_ckpt_name,
                                          every_n_train_steps=model_hparams.ckpt_freq, save_last=True)
    tqdm_pb = NoSmoothingTQDM()
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_callback, tqdm_pb, lr_monitor]  # , device_stats]
    if hparams.device_stats:
        callbacks.append(DeviceStatsMonitor(cpu_stats=True))

    precision = 16 if hparams.fp16 else 32
    trainer = Trainer(gpus=hparams.gpus, profiler="simple", max_epochs=hparams.max_epochs,
                      max_steps=hparams.max_steps if hparams.max_steps != 0 else -1,
                      gradient_clip_val=hparams.gradient_clip_val, callbacks=callbacks,
                      log_every_n_steps=hparams.log_every_n_steps,
                      logger=tb_logger, strategy=hparams.accelerator,
                      detect_anomaly=hparams.detect_anomaly, precision=precision,
                      default_root_dir=root_dir / model_hparams.default_ckpt_root,
                      track_grad_norm=hparams.track_grad_norm)
    restore_path = root_dir / model_hparams.ckpt_dir / model_hparams.restore_ckpt \
        if model_hparams.restore_ckpt is not None and hparams.restore_training else None
    restore_path = restore_path if restore_path is not None and os.path.exists(restore_path) else None

    # supress rank_zero_only/sync_dist warning when logging
    warnings.filterwarnings("ignore", message=".*sync_dist")
    trainer.fit(model, train_dataloaders=train, val_dataloaders=test, ckpt_path=restore_path)


def get_last_path(main_dir, ckpt_dir, best_ckpt):
    file = Path(main_dir) / ckpt_dir / best_ckpt
    return file if os.path.exists(file) else None


def save_hparams(root_path, hparams, filename):
    whole_save = dict(cli_args=sys.argv, hparams=hparams)
    with open(root_path / filename, 'w') as f:
        json.dump(whole_save, fp=f, default=lambda obj: obj.__dict__, indent=4)


def create_logger(root_dir, hparams, level=None, hparams_file="hparams.json"):
    root_dir = Path(root_dir)
    root_dir = root_dir / str(level) if hparams.model == "upsampler" else root_dir
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
    return logger, root_dir
