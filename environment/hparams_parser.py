import argparse
from hparams import default_hparams


help_dictionary = {
    "model": "type of model: 'prior' or 'vqvae' pr 'upsampler'"
}


def hparams_as_parser(parser, hparams):
    hparams.iter(lambda key, value:
                 parser.add_argument(f"--{key}", dest=key, default=value,
                                     type=type(value), help=help_dictionary.get(key))
                 )


def update_hparams_from_parser(default_param, new_hparam):
    return default_param.iter(lambda key, value: new_hparam[key], modify=True)


def create_hparams():
    parser = argparse.ArgumentParser(description='Music generator configuration')
    hparams_as_parser(parser, default_hparams)
    new_params = parser.parse_args().__dict__
    hparams = update_hparams_from_parser(default_hparams, new_params)
    return hparams