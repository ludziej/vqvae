import torch
from pytorch_lightning import LightningModule
from performer_pytorch import AutoregressiveWrapper, PerformerLM
from vqvae.model import VQVAE


class LevelGenerator(LightningModule):
    def __init__(self, vqvae: VQVAE, level: int, log_sample_size: int, num_tokens: int, context_on_level: int,
                 dim: int, depth: int, heads: int, max_seq_len: int, lr: float, start_gen_sample_len: int,
                 log_temperature: float, log_starting_context_len: int, log_context_time: float,
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
        self.context_on_level = context_on_level
        self.log_temperature = log_temperature
        self.log_starting_context_len = log_starting_context_len
        self.log_context_time = log_context_time

    def forward(self, sound: torch.Tensor, context=None) -> torch.Tensor:
        in_tokens, context = self.get_input_and_context(sound, context)
        loss = self.autoregressive.forward(in_tokens) if context is None else \
            self.autoregressive.forward(in_tokens, context=context)
        return loss

    def get_input_and_context(self, sound, context=None):
        endlevel = self.level + 1 if not self.context_on_level else self.level + 2
        in_tokens = [x.detach() for x in self.preprocessing.encode(sound, start_level=self.level, end_level=endlevel)]
        in_tokens, context = in_tokens if self.context_on_level else (in_tokens[0], context)
        return in_tokens, context

    def recreate_beginning(self, bs=None):
        bs = 1 if bs is None else bs
        return torch.randint(self.num_tokens, (bs, self.start_gen_sample_len, 1), device=self.device)

    # TODO fast wrapper
    # this probably has still quadratic complexity
    def generate(self, seq_len: int, beginning=None, temperature=1., bs=None, context=None):
        with torch.no_grad():
            beginning = self.recreate_beginning(bs) if beginning is None else beginning
            #seq_len = self.preprocessing.downsample_level(seq_len, level=self.level)  # Not implemented
            out_tokens = self.autoregressive.generate(beginning, seq_len=seq_len, temperature=temperature, context=context)
            sound = self.preprocessing.decode([out_tokens], start_level=self.level, end_level=self.level + 1).squeeze(2)
            return sound

    def generate_from_sound(self, sound: torch.Tensor, prefix_token_perc: float, temperature=1., context=None):
        in_tokens, context = self.get_input_and_context(sound, context)
        org_len = in_tokens.shape[1]
        pref_len = int(org_len * prefix_token_perc)
        in_tokens = in_tokens[:, pref_len:]
        return self.generate(org_len - pref_len, beginning=in_tokens, context=context, temperature=temperature)

    def log_metrics_and_samples(self, loss, batch, batch_idx, prefix=""):
        nr = self.log_nr.get(prefix, 0)
        self.log_nr[prefix] = nr + 1
        self.log(prefix + "loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if batch_idx != 0:
            return  # log samples once per epoch
        tlogger = self.logger.experiment

        # generate continuation audio
        con_samples = self.generate_from_sound(batch[:, :self.log_context_time], self.log_starting_context_len,
                                               temperature=self.log_temperature)
        for i, sample in enumerate(con_samples):
                tlogger.add_audio(prefix + f"sample_con_{i}", sample, nr, self.sr)

        if not self.context_on_level and prefix == "":  # raw only for train, because it does not depend on input data
            samples = self.generate(self.log_sample_size, bs=self.log_sample_bs, temperature=self.log_temperature)
            for i, sample in enumerate(samples):
                tlogger.add_audio(prefix + f"sample_raw_{i}", sample, nr, self.sr)

    def training_step(self, batch, batch_idx, name=""):
        loss = self(batch)
        self.log_metrics_and_samples(loss, batch, batch_idx, name)
        return loss

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, "test_")

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, "val_")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        # sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
        # return [opt], [sched]
        return opt


class SynchrGenerator(torch.nn.Module):
    def __init__(self, vqvae: VQVAE, prior: LevelGenerator, upsamplers: [LevelGenerator]):
        super().__init__()
        self.vqvae = vqvae
        self.prior = prior
        self.upsamplers = torch.nn.ModuleList(upsamplers)

    def generate(self, time: float) -> torch.Tensor:
        raise Exception("Not Implemented")


