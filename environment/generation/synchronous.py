import torch
import torch.nn as nn

from environment.generation.token_generator import SynchronousTokenGenerator
from vqvae.model import VQVAE
from generator.model import LevelGenerator, GenerationParams
from data_processing.tools import load_file, save_file


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

    def continue_sound(self, sound: torch.Tensor, sec_from: int, params: GenerationParams,
                       use_tqdm=True, decode_level=0,) -> torch.Tensor:
        sound = sound[:sec_from * self.vqvae.sr]
        added_seconds = params.time - sec_from
        add_tokens_length = self.vqvae.samples_num_to_tokens(self.vqvae.sr * added_seconds)
        encoded = self.encode_sound(sound)
        new_tokens = self.tokens_gen.continuation(encoded, add_tokens_length, use_tqdm=use_tqdm, params=params,
                                                  decode_level=decode_level)
        decoded = self.decode_sound(new_tokens, decode_level)
        return decoded

    def generate_sound(self, time: float, params: GenerationParams, bs: int = 1, decode_level=0, use_tqdm=True)\
        -> torch.Tensor:
        tokens_length = self.vqvae.samples_num_to_tokens(self.vqvae.sr * time)
        tokens = self.tokens_gen.generate(tokens_length, params, bs, decode_level, use_tqdm)
        decoded = self.decode_sound(tokens, decode_level)
        return decoded

    # file operations

    def load_file(self, filenames):
        if isinstance(filenames, str):
            return load_file(filenames, sr=self.vqvae.sr)
        return torch.stack([self.load_file(f) for f in filenames])

    def save_file(self, data, filenames):
        filenames = [f"{filenames}_{i}.wav" for i in range(data.shape[0])]\
            if isinstance(filenames, str) else [f + ".wav" for f in filenames]
        for f, d in zip(filenames, data):
            return save_file(d, f, sr=self.vqvae.sr)

    # file ops

    def get_track_through_vqvae(self, filepaths: str, out_path: str, level=0, **params):
        sound = self.get_sound_through_vqvae(self.load_file(filepaths), level=level)
        self.save_file(sound, out_path)

    def continue_track(self, filepaths: str, out_path: str, gen_params: GenerationParams, sec_from: int,
                       use_tqdm=True, **params):
        sound = self.continue_sound(self.load_file(filepaths), sec_from, use_tqdm=use_tqdm, params=gen_params)
        self.save_file(sound, out_path)

    def generate_trach(self, out_path: str, time: float, gen_params: GenerationParams, bs: int = 1, decode_level=0,
                       use_tqdm=True, **params):
        sound = self.generate_sound(time=time, params=gen_params, bs=bs, decode_level=decode_level, use_tqdm=use_tqdm)
        self.save_file(sound, out_path)

    # helpers

    # TODO implement properly when after conditioning is implemented
    def resolve_generation_params(self, artist, time, bpm, sec_from=None) -> GenerationParams:
        artist = artist if artist != -1 else 0  # TODO randomize
        time = time if time != -1 else 240  # TODO randomize over_sec_from, some distr
        bpm = None  # TODO some random distr
        return GenerationParams(artist=artist, time=time)
