from hparams.parser import HparamsParser
from hparams.config import hparams_registry
from environment.generation.synchronous import SynchronousGenerator
from generator.modules.conditioner import GenerationParams
from environment.train_transformer import get_model as get_transformer
from environment.train_embeddings import get_model as get_vqvae
from environment.train_utils import create_logger
from pathlib import Path


legal_generation_configs = ["gen_big"]
action_mapping = {
    "encdec": SynchronousGenerator.get_track_through_vqvae,
    "continue": SynchronousGenerator.continue_track,
    "generate": SynchronousGenerator.generate_trach,
}


def create_generator(hparams) -> SynchronousGenerator:
    logger, root_dir = create_logger(hparams.logger_dir, hparams)
    vqvae = get_vqvae(logger=logger, **hparams.vqvae, with_train_data=False, train_path=hparams.train_path)
    prior = get_transformer(vqvae=vqvae, logger=logger, level=len(hparams.upsampler), **hparams.prior)
    upsamplers = [get_transformer(vqvae=vqvae, logger=logger, level=level, **hp)
                  for level, hp in enumerate(hparams.upsampler)]
    return SynchronousGenerator(vqvae=vqvae, prior=prior, upsamplers=upsamplers)


def get_gen_params(generator: SynchronousGenerator, artist, time, bpm, sec_from, **params) -> GenerationParams:
    return generator.resolve_generation_params(artist, time, bpm, sec_from)


def take_action(generator: SynchronousGenerator, action, **params):
    params = dict(gen_params=get_gen_params(generator, **params), **params)
    if action not in action_mapping:
        raise Exception(f"Unknown action {action}, leval action are {action_mapping.keys()}")
    return action_mapping[action](generator, **params)


def run_generation(params):
    generator = create_generator(params)
    return take_action(generator, **params)


def run():
    hparams = HparamsParser(hparams_registry, default_hparams="gen_big").create_hparams()
    if "action" not in hparams:
        raise Exception(f"Wrong config selected, for generation only {legal_generation_configs} are available")
    return run_generation(hparams)


if __name__ == "__main__":
    run()
