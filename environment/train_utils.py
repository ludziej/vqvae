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
from utils.misc import exists


class NoSmoothingTQDM(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.smoothing = 0
        return bar


def get_neptune_hparams(model_type):
    return dict(
            project=f"wavefusion/{model_type}",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzNDYxZjgzZC0xYTljLTQwZGQtOTVjNC02MTI5ZTc4ZjBiNGIifQ==",
        )


def try_get_last_from_neptune(last_path, neptune_path, neptune_run_id, neptune_project, logger):
    if neptune_path == "" or neptune_run_id == "" or neptune_project == "" or os.path.exists(last_path):
        return last_path if os.path.exists(last_path) else None
    import neptune
    run = neptune.init_run(**get_neptune_hparams(neptune_project), mode="read-only", with_id=neptune_run_id)
    logger.info(f"Downloading model from {neptune_project}/{neptune_run_id}/{neptune_path} to {last_path}...")
    last_path.parent.mkdir(parents=True, exist_ok=True)
    run[neptune_path].download(str(last_path))
    run.sync(wait=True)
    run.stop()
    logger.info(f"Downloaded model from neptune")
    return last_path


def generic_get_model(name, model_class, main_dir, ckpt_dir, restore_ckpt, logger_type, logger,
                      neptune_path="", neptune_run_id="", neptune_project="", **params):
    last_path = get_last_path(main_dir, ckpt_dir, restore_ckpt, return_non_existing=True)
    last_path = try_get_last_from_neptune(last_path, neptune_path, neptune_run_id, neptune_project, logger)

    logger.info(f"Restoring {name} from {last_path}" if last_path else f"Starting {name} training from scratch")
    params.update(dict(logger=logger, logger_type=logger_type))
    model = model_class.load_from_checkpoint(last_path, **params, strict=False) \
        if last_path is not None else model_class(**params)
    params["logger"].debug(f"Model {name} loaded")
    return model


def get_logger(root_dir, hparams, model_hparams):
    if model_hparams.logger_type == "tensorboard":
        return pl_loggers.TensorBoardLogger(root_dir / model_hparams.logs_dir)
    elif model_hparams.logger_type == "neptune":
        return pl_loggers.NeptuneLogger(**get_neptune_hparams(model_hparams.model_type))
    else:
        raise Exception(f"Unknown logger type {hparams.logger_type}")


def generic_train(model, hparams, train, test, model_hparams, root_dir):
    logger = get_logger(root_dir, hparams, model_hparams)
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
                      logger=logger, strategy=hparams.accelerator,
                      detect_anomaly=hparams.detect_anomaly, precision=precision,
                      default_root_dir=root_dir / model_hparams.default_ckpt_root,
                      track_grad_norm=hparams.track_grad_norm,
                      check_val_every_n_epoch=hparams.check_val_every_n_epoch)
    restore_path = root_dir / model_hparams.ckpt_dir / model_hparams.restore_ckpt \
        if model_hparams.restore_ckpt is not None and hparams.restore_training else None
    restore_path = restore_path if restore_path is not None and os.path.exists(restore_path) else None

    # supress rank_zero_only/sync_dist warning when logging
    warnings.filterwarnings("ignore", message=".*sync_dist")
    trainer.fit(model, train_dataloaders=train, val_dataloaders=test, ckpt_path=restore_path)


def get_last_path(main_dir, ckpt_dir, best_ckpt, return_non_existing=False):
    file = Path(main_dir) / ckpt_dir / best_ckpt
    return file if return_non_existing or os.path.exists(file) else None


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
