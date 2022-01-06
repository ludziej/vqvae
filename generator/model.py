import torch
from pytorch_lightning import LightningModule
from performer_pytorch import AutoregressiveWrapper, PerformerLM
from vqvae.model import VQVAE


class Prior(LightningModule):
    def __init__(self, vqvae: VQVAE, level: int, log_sample_size: int, num_tokens: int,
                 dim: int, depth: int, heads: int, max_seq_len: int, lr: float, start_gen_sample_len: int,
                 **kwargs):
        super().__init__()
        self.level = level
        self.num_tokens = num_tokens
        self.log_sample_bs, self.log_sample_size = log_sample_size
        self.preprocessing = vqvae
        self.sr = vqvae.sr
        self.preprocessing.freeze()
        self.start_gen_sample_len = start_gen_sample_len
        self.lr = lr
        self.transformer = PerformerLM(causal=True, num_tokens=num_tokens, dim=dim, depth=depth, heads=heads,
                                       max_seq_len=max_seq_len)
        self.autoregressive = AutoregressiveWrapper(net=self.transformer)
        self.log_nr = {"val_": 0, "": 0, "test_": 0}

    def forward(self, sound: torch.Tensor, context=None, context_on_level=False) -> torch.Tensor:
        in_tokens, context = self.get_input_and_context(sound, context, context_on_level)
        y_tokens = self.autoregressive.forward(in_tokens) if context is None else \
            self.autoregressive.forward(in_tokens, context=context)
        return y_tokens

    def get_input_and_context(self, sound, context=None, context_on_level=False):
        endlevel = self.level + 1 if not context_on_level else self.level + 2
        in_tokens = [x.detach() for x in self.preprocessing.encode(sound, start_level=self.level, end_level=endlevel)]
        in_tokens, context = in_tokens if context_on_level else (in_tokens[0], context)
        return in_tokens, context

    def recreate_beginning(self, bs=None):
        bs = 1 if bs is None else bs
        return torch.randint(1, (bs, self.start_gen_sample_len, 1), device=self.device)

    # TODO fast wrapper
    # this probably has still quadratic complexity
    # CONTEXT SUPPORT
    def generate(self, seq_len: int, beginning=None, temperature=1., bs=None):
        with torch.no_grad():
            beginning = self.recreate_beginning(bs) if beginning is None else beginning
            prev_tokens = self.preprocessing.encode(beginning, start_level=self.level, end_level=self.level + 1)[0]
            #seq_len = self.preprocessing.downsample_level(seq_len, level=self.level)  # Not implemented
            out_tokens = self.autoregressive.generate(prev_tokens, seq_len=seq_len, temperature=temperature)
            sound = self.preprocessing.decode([out_tokens], start_level=self.level, end_level=self.level + 1).squeeze(2)
            return sound

    def log_metrics_and_samples(self, loss, batch_idx, prefix=""):
        nr = self.log_nr.get(prefix, 0)
        self.log_nr[prefix] = nr + 1
        self.log(prefix + "loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if batch_idx != 0:
            return  # log samples once per epoch
        tlogger = self.logger.experiment
        samples = self.generate(self.log_sample_size, bs=self.log_sample_bs)
        for i, sample in enumerate(samples):
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

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        # sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
        # return [opt], [sched]
        return opt
