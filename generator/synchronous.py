import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from performer_pytorch import PerformerLM
from vqvae.model import VQVAE
import torch.nn.functional as F
from generator.model import LevelGenerator


class SynchronousGenerator(nn.Module):
    """Class for generation, when all models are properly trained"""
    def __init__(self, vqvae: VQVAE, prior: LevelGenerator, upsamplers: [LevelGenerator]):
        super().__init__()
        self.vqvae = vqvae
        self.prior = prior
        self.upsamplers = torch.nn.ModuleList(upsamplers)

    # generation

    def generate_prior_tokens(self, length, bs=1, with_tqdm=True, context=None):
        return self.prior.generate_no_sound(length, context=context, with_tqdm=with_tqdm, bs=bs)

    def generate_upsampler_tokens(self, length, prev_tokens, level, with_tqdm=True, context=None):
        upsampler = self.upsamplers[level]
        return upsampler.generate_no_sound(length, context=context, with_tqdm=with_tqdm, bs=prev_tokens.shape[0])

    def generate(self, time: float, bs: int = 1, decode_level = 0, with_tqdm=True, context=None, time_context=None)\
            -> torch.Tensor:
        params = dict(with_tqdm=with_tqdm, context=context, time=time_context)
        tokens_length = [None, None, None]  # TODO get from "time"
        tokens = self.generate_prior_tokens(tokens_length[-1], bs=bs, **params)
        for level in reversed(range(decode_level, self.prior.level)):
            tokens = self.generate_upsampler_tokens(tokens_length[level], tokens, level, **params)
        sound = self.decode_sound(tokens, decode_level)
        return sound

    # vqvae operations

    # assumes sound is with correct sr = self.vqvae.sr
    def get_sound_through_vqvae(self, sound: torch.Tensor):
        raise Exception("Not Implemented")

    def get_track_through_vqvae(self, filepath: str):
        raise Exception("Not Implemented")

    def decode_sound(self, tokens, level):
        raise Exception("Not Implemented")

    def encode_sound(self, sound, level):
        raise Exception("Not Implemented")

    # continuation

    # assumes sound is with correct sr = self.vqvae.sr
    def continue_sound(self, sound: torch.Tensor, use_tqdm=True) -> torch.Tensor:
        raise Exception("Not Implemented")

    def continue_track(self, filepath: str, sec_from: int, use_tqdm=True) -> torch.Tensor:
        raise Exception("Not Implemented")
