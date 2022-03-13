import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from performer_pytorch import PerformerLM
from vqvae.model import VQVAE
import torch.nn.functional as F
from generator.model import LevelGenerator


class SynchronousGenerator(nn.Module):
    def __init__(self, vqvae: VQVAE, prior: LevelGenerator, upsamplers: [LevelGenerator]):
        super().__init__()
        self.vqvae = vqvae
        self.prior = prior
        self.upsamplers = torch.nn.ModuleList(upsamplers)

    def generate_prior_tokens(self):
        raise Exception("Not Implemented")

    def generate_upsampler_tokens(self, prev_tokens):
        raise Exception("Not Implemented")

    def get_sound_through_vqvae(self, sound: torch.Tensor):
        raise Exception("Not Implemented")

    def get_track_through_vavae(self, filepath: str):
        raise Exception("Not Implemented")

    def generate(self, time: float, bs: int, decode_level=0) -> torch.Tensor:
        raise Exception("Not Implemented")

    # assumes sound with correct sr = self.vqvae.sr
    def continue_sound(self, sound: torch.Tensor) -> torch.Tensor:
        raise Exception("Not Implemented")

    def continue_track(self, filepath: str, sec_from: int) -> torch.Tensor:
        raise Exception("Not Implemented")
