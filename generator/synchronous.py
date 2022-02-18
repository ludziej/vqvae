import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from performer_pytorch import PerformerLM
from vqvae.model import VQVAE
import torch.nn.functional as F
from generator.model import LevelGenerator


class SynchronousGenerator(LightningModule):
    def __init__(self, vqvae: VQVAE, prior: LevelGenerator, upsamplers: [LevelGenerator]):
        super().__init__()
        self.vqvae = vqvae
        self.prior = prior
        self.upsamplers = torch.nn.ModuleList(upsamplers)

    def generate(self, time: float) -> torch.Tensor:
        raise Exception("Not Implemented")

