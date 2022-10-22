import torch
from torch import nn as nn

from generator.model import LevelGenerator, GenerationParams


class SynchronousTokenGenerator(nn.Module):
    """Class for tokens generation, when all models are properly trained"""
    def __init__(self, prior: LevelGenerator, upsamplers: [LevelGenerator]):
        super().__init__()
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
        raise Exception("Not Implemented")  # TODO

    def continue_upsampler(self, previous, length, up_tokens, params: GenerationParams, use_tqdm=True):
        raise Exception("Not Implemented")  # TODO

    # synchronous token operations

    def generate(self, tokens_length: [int], params: GenerationParams, bs: int = 1, decode_level=0, with_tqdm=True)\
            -> torch.Tensor:
        tokens = self.generate_prior(tokens_length[-1], bs=bs, params=params, with_tqdm=with_tqdm)
        for level in reversed(range(decode_level, self.prior.level)):
            tokens = self.generate_upsampler(tokens_length[level], tokens, level,
                                             params=params, with_tqdm=with_tqdm)
        return tokens

    def continuation(self, encoded: [torch.Tensor], add_tokens_length: [int], params: GenerationParams,
                     use_tqdm=True, decode_level=0,) -> torch.Tensor:
        tokens = self.continue_prior(encoded[-1], length=add_tokens_length[-1],
                                     params=params, use_tqdm=use_tqdm)
        for level in reversed(range(decode_level, self.prior.level)):
            tokens = self.continue_upsampler(encoded[level], add_tokens_length[level], tokens,
                                             params=params, use_tqdm=use_tqdm)
        return tokens
