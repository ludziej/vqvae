from hparams.parser import HparamsParser
from hparams.config import hparams_registry
from environment.generation.synchronous import SynchronousGenerator, GenerationParams
from environment.train_transformer import get_model as get_transformer
from environment.train_embeddings import get_model as get_vqvae
from environment.train_utils import create_logger


legal_generation_configs = ["gen_big"]
action_mapping = {
    "encdec": SynchronousGenerator.get_track_through_vqvae,
    "continue": SynchronousGenerator.continue_track,
    "generate": SynchronousGenerator.generate_trach,
}


def create_generator(hparams) -> SynchronousGenerator:
    logger = create_logger(hparams.logger_dir, hparams)
    vqvae = get_vqvae(logger=logger, **hparams.vqvae, with_train_data=False)
    prior = get_transformer(vqvae=vqvae, logger=logger, level=len(hparams.upsamplers), **hparams.prior)
    upsamplers = [get_transformer(vqvae=vqvae, logger=logger, level=level, **hp)
                  for level, hp in enumerate(hparams.upsamplers)]
    return SynchronousGenerator(vqvae=vqvae, prior=prior, upsamplers=upsamplers)


def get_gen_params(generator: SynchronousGenerator, artist, time, bpm, sec_from, **params) -> GenerationParams:
    return generator.resolve_generation_params(artist, time, bpm, sec_from)


def take_action(generator: SynchronousGenerator, action, **params):
    params = {**get_gen_params(**params), **params}
    if action not in action_mapping:
        raise Exception(f"Unknown action {action}, leval action are {action_mapping.keys()}")
    return action_mapping[action](generator, **params)


def run_generation(**params):
    generator = create_generator(params, **params)
    return take_action(generator, **params)


def run():
    hparams = HparamsParser(hparams_registry).create_hparams()
    if "operation" not in hparams:
        raise Exception(f"Wrong config selected, for generation available are {legal_generation_configs}")
    return run_generation(**hparams)


if __name__ == "__main__":
    run()
