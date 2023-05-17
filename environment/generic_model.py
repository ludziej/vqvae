from pytorch_lightning import LightningModule
import torch
import abc


class GenericModel(LightningModule, abc.ABC):
    def __init__(self, **params):
        super().__init__()
        self.train_dl_fun = None
        self.val_dl_fun = None
        self.get_audio_logger = None
        self.is_first_log = True
        for name, val in params.items():
            setattr(self, name, val)

    def set_dataloaders(self, train_dl, val_dl):
        self.train_dl_fun = train_dl
        self.val_dl_fun = val_dl

    @property
    def audio_logger(self):
        assert self.get_audio_logger is not None  # set_audio_logger must be called before
        return self.get_audio_logger()

    def set_audio_logger(self, logger):
        self.get_audio_logger = logger

    def on_after_backward(self) -> None:
        self.audio_logger.log_grads()

    def train_dataloader(self):
        assert self.train_dl_fun is not None and self.batch_size != 0
        return self.train_dl_fun(self.batch_size)

    def val_dataloader(self):
        assert self.val_dl_fun is not None and self.batch_size != 0
        return self.val_dl_fun(self.batch_size)

    def should_run_heavy_logs(self, batch_idx, phase, **params):
        return batch_idx % self.log_interval == 0 and self.local_rank == 0 and\
               (phase == "train" or not self.skip_valid_logs)

    def training_step(self, batch, batch_idx, **params):
        return self.evaluation_step(batch, batch_idx, "train", **params)

    def test_step(self, batch, batch_idx, **params):
        return self.evaluation_step(batch, batch_idx, "test", **params)

    def validation_step(self, batch, batch_idx, **params):
        return self.evaluation_step(batch, batch_idx, "val", **params)

    @abc.abstractmethod
    def evaluation_step(self, batch, batch_idx, phase="train", **params):
        """"""

    def heavy_logs(self, **params):
        self.is_first_log = False
        self.audio_logger.switch_full_grad_log()

    def light_logs(self, metrics, prefix, loss, **params):
        self.audio_logger.log_metrics({**metrics, 'loss': loss}, prefix)

    @torch.no_grad()
    def log_metrics_and_samples(self, batch_idx, phase, **params):
        prefix = phase + "_" if phase != "train" else ""
        self.light_logs(prefix=prefix, **params)
        if self.should_run_heavy_logs(batch_idx=batch_idx, phase=phase, **params):
            self.heavy_logs(prefix=prefix, **params)
