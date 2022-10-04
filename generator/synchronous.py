import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from performer_pytorch import PerformerLM
from vqvae.model import VQVAE
import torch.nn.functional as F
from generator.model import LevelGenerator
from data_processing.tools import load_file, save_file


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

    def generate(self, time: float, bs: int = 1, decode_level=0, with_tqdm=True, context=None, time_context=None)\
            -> torch.Tensor:
        params = dict(with_tqdm=with_tqdm, context=context, time=time_context)
        tokens_length = self.vqvae.get_z_lengths(self.vqvae.sr * time)
        tokens = self.generate_prior_tokens(tokens_length[-1], bs=bs, **params)
        for level in reversed(range(decode_level, self.prior.level)):
            tokens = self.generate_upsampler_tokens(tokens_length[level], tokens, level, **params)
        sound = self.decode_sound(tokens, decode_level)
        return sound

    # vqvae operations

    # assumes sound is with correct sr = self.vqvae.sr
    def get_sound_through_vqvae(self, sound: torch.Tensor, level=0):
        encoded = self.encode_sound(sound, level, level + 1)[0]
        return self.decode_sound(encoded, level)

    def decode_sound(self, tokens, level):
        return self.preprocessing.decode([tokens], start_level=level, end_level=level + 1).squeeze(2).detach()

    def encode_sound(self, sound, start_level=0, end_level=None):
        return [x.detach() for x in self.preprocessing.encode(sound, start_level=start_level,
                                                              end_level=end_level or self.prior.level + 1)]

    # continuation

    def continue_prior_tokens(self, previous, length, context=None, time=None):
        raise Exception("Not Implemented")

    def continue_upsampler_tokens(self, previous, length, context=None, time=None):
        raise Exception("Not Implemented")

    # assumes sound is with correct sr = self.vqvae.sr
    def continue_sound(self, sound: torch.Tensor, sec_from: int, added_seconds: int, use_tqdm=True) -> torch.Tensor:
        sound = sound[:sec_from * self.vqvae.sr]
        add_tokens_length = self.vqvae.get_z_lengths(self.vqvae.sr * added_seconds)
        prior = self.prior.generate_from_sound(sound, prefix_token_perc=1, use_tqdm=use_tqdm)



    # file operations

    def load_file(self, filename):
        return load_file(filename, sr=self.vqvae.sr)

    def save_file(self, filename):
        return save_file(filename, sr=self.vqvae.sr)

    def continue_track(self, filepath: str, sec_from: int, added_seconds: int, use_tqdm=True) -> torch.Tensor:
        sound = self.load_file(filepath)
        sound = self.continue_sound(sound, sec_from, added_seconds, use_tqdm=use_tqdm)
        return sound

    def get_track_through_vqvae(self, filepath: str, level=0):
        return self.get_sound_through_vqvae(self.load_file(filepath), level=level)
