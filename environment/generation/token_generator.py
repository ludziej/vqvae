import torch
from torch import nn as nn

from generator.model import LevelGenerator
from generator.modules.conditioner import GenerationParams


class SynchronousTokenGenerator(nn.Module):
    """Class for tokens generation, when all models are properly trained"""
    def __init__(self, prior: LevelGenerator, upsamplers: [LevelGenerator]):
        super().__init__()
        self.prior = prior
        self.upsamplers = torch.nn.ModuleList(upsamplers)

    # generation

    def generate_prior(self, length, params: GenerationParams, bs=1, with_tqdm=True):
        return self.prior.generate_tokens(length, param=params, with_tqdm=with_tqdm, bs=bs)

    def generate_upsampler(self, length, prev_tokens, level, params: GenerationParams, with_tqdm=True):
        return self.upsamplers[level].generate_tokens(length, param=params, with_tqdm=with_tqdm,
                                                      bs=prev_tokens.shape[0], up_tokens=prev_tokens)

    def continue_prior(self, previous, length, params: GenerationParams, with_tqdm=True, with_begin=True):
        return self.prior.continue_tokens(tokens=previous, prefix_token_perc=1, with_tqdm=with_tqdm,
                                          length=previous.shape[-1] + length, params=params, with_begin=with_begin)[0]

    def continue_upsampler(self, previous, length, up_tokens, level, params: GenerationParams, with_tqdm=True,
                           with_begin=True):
        return self.upsamplers[level].continue_tokens(tokens=previous, prefix_token_perc=1, with_tqdm=with_tqdm,
                                                      length=previous.shape[-1] + length, params=params,
                                                      up_tokens=up_tokens, with_begin=with_begin)[0]

    # synchronous token operations

    def generate(self, tokens_length: [int], params: GenerationParams, bs: int = 1, decode_level=0, with_tqdm=True)\
            -> torch.Tensor:
        tokens = self.generate_prior(tokens_length[-1], bs=bs, params=params, with_tqdm=with_tqdm)
        for level in reversed(range(decode_level, self.prior.level)):
            tokens = self.generate_upsampler(tokens_length[level], tokens, level,
                                             params=params, with_tqdm=with_tqdm)
        return tokens

    def continuation(self, encoded: [torch.Tensor], add_tokens_length: [int], params: GenerationParams,
                     with_tqdm=True, decode_level=0,) -> torch.Tensor:
        tokens = self.continue_prior(encoded[-1], length=add_tokens_length[-1],
                                     params=params, with_tqdm=with_tqdm)
        for level in reversed(range(decode_level, self.prior.level)):
            tokens = self.continue_upsampler(encoded[level], add_tokens_length[level], tokens, level=level,
                                             params=params, with_tqdm=with_tqdm)
        return tokens
