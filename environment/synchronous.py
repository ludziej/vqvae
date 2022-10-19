import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from performer_pytorch import PerformerLM
from vqvae.model import VQVAE
import torch.nn.functional as F
from generator.model import LevelGenerator, GenerationParams
from data_processing.tools import load_file, save_file


class SynchronousTokenGenerator(nn.Module):
    """Class for tokens generation, when all models are properly trained"""
    def __init__(self, vqvae: VQVAE, prior: LevelGenerator, upsamplers: [LevelGenerator]):
        super().__init__()
        self.vqvae = vqvae
        self.prior = prior
        self.upsamplers = torch.nn.ModuleList(upsamplers)

    # generation

    def generate_prior(self, length, params: GenerationParams, bs=1, with_tqdm=True):
        raise Exception("Not Implemented") # TODO
        return self.prior.generate_no_sound(length, context=context, with_tqdm=with_tqdm, bs=bs)

    def generate_upsampler(self, length, prev_tokens, level, params: GenerationParams, with_tqdm=True):
        upsampler = self.upsamplers[level]
        raise Exception("Not Implemented") # TODO
        return upsampler.generate_no_sound(length, context=context, with_tqdm=with_tqdm, bs=prev_tokens.shape[0])

    def continue_prior(self, previous, length, params: GenerationParams, use_tqdm=True):
        prior = self.prior.generate_from_sound(sound, prefix_token_perc=1, use_tqdm=use_tqdm)
        raise Exception("Not Implemented") # TODO

    def continue_upsampler(self, previous, length, up_tokens, params: GenerationParams, use_tqdm=True):
        raise Exception("Not Implemented") # TODO

    # synchronous token operations

    def generate(self, time: float, params: GenerationParams, bs: int = 1, decode_level=0, with_tqdm=True)\
            -> torch.Tensor:
        tokens_length = self.vqvae.samples_num_to_tokens(self.vqvae.sr * time)
        tokens = self.generate_prior(tokens_length[-1], bs=bs, params=params, with_tqdm=with_tqdm)
        for level in reversed(range(decode_level, self.prior.level)):
            tokens = self.generate_upsampler(tokens_length[level], tokens, level,
                                             params=params, with_tqdm=with_tqdm)
        return tokens

    def continuation(self, encoded: [torch.Tensor], add_tokens_length: [int], params: GenerationParams,
                        use_tqdm=True, decode_level=0,) -> torch.Tensor:
        tokens = self.continue_prior(encoded[-1], length=add_tokens_length[-1],
                                     params=params, use_tqdm=use_tqdm)
        all_tokens = [tokens]
        for level in reversed(range(decode_level, self.prior.level)):
            tokens = self.continue_upsampler(encoded[level], add_tokens_length[level], tokens,
                                             params=params, use_tqdm=use_tqdm)
            all_tokens.append(tokens)
        return tokens


class SynchronousGenerator(nn.Module):
    """Class for sound generation, when all models are properly trained"""
    def __init__(self, vqvae: VQVAE, prior: LevelGenerator, upsamplers: [LevelGenerator]):
        super().__init__()
        self.vqvae = vqvae
        self.prior = prior
        self.upsamplers = torch.nn.ModuleList(upsamplers)
        self.tokens_gen = SynchronousTokenGenerator(vqvae, prior, upsamplers)

    # vqvae operations

    def decode_sound(self, tokens, level):
        return self.preprocessing.decode([tokens], start_level=level, end_level=level + 1).squeeze(2).detach()

    def encode_sound(self, sound, start_level=0, end_level=None):
        return [x.detach() for x in self.preprocessing.encode(sound, start_level=start_level,
                                                              end_level=end_level or self.prior.level + 1)]

    # assumes sound is with correct sr = self.vqvae.sr
    def get_sound_through_vqvae(self, sound: torch.Tensor, level=0):
        encoded = self.encode_sound(sound, level, level + 1)[0]
        return self.decode_sound(encoded, level)

    # sound operation wrappers
    # assumes sound is with correct sr = self.vqvae.sr

    def continue_sound(self, sound: torch.Tensor, sec_from: int, added_seconds: int, params: GenerationParams,
                       use_tqdm=True, decode_level=0,) -> torch.Tensor:
        sound = sound[:sec_from * self.vqvae.sr]
        add_tokens_length = self.vqvae.samples_num_to_tokens(self.vqvae.sr * added_seconds)
        encoded = self.encode_sound(sound)
        new_tokens = self.tokens_gen.continuation(encoded, add_tokens_length, use_tqdm=use_tqdm, params=params,
                                                  decode_level=decode_level)
        decoded = self.decode_sound(new_tokens, decode_level)
        return decoded

    def generate_sound(self, time: float, params: GenerationParams, bs: int = 1, decode_level=0, with_tqdm=True)\
        -> torch.Tensor:
        tokens = self.tokens_gen.generate(time, params, bs, decode_level, with_tqdm)
        decoded = self.decode_sound(tokens, decode_level)
        return decoded

    # file operations

    def load_file(self, filename):
        return load_file(filename, sr=self.vqvae.sr)

    def save_file(self, filename):
        return save_file(filename, sr=self.vqvae.sr)

    def continue_track(self, filepath: str, params: GenerationParams, sec_from: int, added_seconds: int,
                       use_tqdm=True) -> torch.Tensor:
        sound = self.load_file(filepath)
        sound = self.continue_sound(sound, sec_from, added_seconds, use_tqdm=use_tqdm, params=params)
        return sound

    def get_track_through_vqvae(self, filepath: str, level=0):
        return self.get_sound_through_vqvae(self.load_file(filepath), level=level)

    # helpers

    def resolve_generation_params(self) -> GenerationParams:
        raise Exception("Not Implemented")
