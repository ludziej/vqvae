import torch
from pytorch_lightning import LightningModule
from performer_pytorch import AutoregressiveWrapper, PerformerLM
from vqvae.model import VQVAE
import torch.nn as nn
import numpy as np


class Prior(LightningModule):
    def __init__(self, vqvae: VQVAE, level: int, log_sample_size: int, num_tokens: int, **kwargs):
        super().__init__()
        self.level = level
        self.num_tokens = num_tokens
        self.log_sample_bs, self.log_sample_size = log_sample_size
        self.preprocessing = vqvae
        self.preprocessing.freeze()
        self.min_beg_length = 10

        self.transformer = PerformerLM(causal=True, num_tokens=num_tokens, **kwargs)
        self.autoregressive = AutoregressiveWrapper(net=self.transformer)
        self.log_nr = {"val_": 0, "": 0, "test_": 0}

    def forward(self, sound: torch.Tensor, context=None, context_on_level=False) -> torch.Tensor:
        in_tokens, context = self.get_input_and_context(sound, context, context_on_level)
        y_tokens = self.autoregressive.forward(in_tokens) if context is None else \
            self.autoregressive.forward(in_tokens, context=context)
        return y_tokens

    def get_input_and_context(self, sound, context=None, context_on_level=False):
        endlevel = self.level + 1 if not context_on_level else self.level + 2
        in_tokens = self.preprocessing.encode(sound, start_level=self.level, end_level=endlevel).detach()
        in_tokens, context = in_tokens if context_on_level else (in_tokens[0], context)
        return in_tokens, context

    def recreate_beginning(self, bs=None):
        bs = 1 if bs is None else bs
        return torch.randint((bs, self.min_beg_length))

    # TODO fast wrapper
    # this probably has still quadratic complexity
    # CONTEXT SUPPORT
    def generate(self, seq_len: float, beginning=None, temperature=1., bs=None):
        with torch.no_grad():
            beginning = self.recreate_beginning(bs) if beginning is None else beginning
            prev_tokens = self.preprocessing.encode(beginning, start_level=self.level, end_level=self.level + 1)[0]
            seq_len_tokens = self.preprocessing.downsample_level(seq_len, level=self.level)
            out_tokens = self.autoregressive.generate(prev_tokens, seq_len=seq_len_tokens, temperature=temperature)
            sound = self.preprocessing.decode(out_tokens, start_level=self.level, end_level=self.level + 1)
            return sound

    def log_metrics_and_samples(self, loss, batch_idx, prefix=""):
        nr = self.log_nr.get(prefix, 0)
        self.log_nr[prefix] = nr + 1
        self.log(prefix + "loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if batch_idx != 0:
            return  # log samples once per epoch
        tlogger = self.logger.experiment
        samples = self.generate(self.log_sample_size, bs=self.log_sample_bs)
        for i, sample in samples:
            tlogger.add_audio(prefix + f"sample_{i}", sample, nr, self.sr)

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log_metrics_and_samples(loss, batch_idx, "")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self(batch)
        self.log_metrics_and_samples(loss, batch_idx, "test_")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.log_metrics_and_samples(loss, batch_idx, "val_")
        return loss