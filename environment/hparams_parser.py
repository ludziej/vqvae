import argparse
from types import SimpleNamespace
from collections.abc import Mapping


def fst(x, y):
    return x


class HparamsParser:

    help_dictionary = {
        "model": "type of model: 'prior' or 'vqvae' pr 'upsampler'"
    }

    type_dict = {int: int, float: float}

    def __init__(self, hparams_dict, top_hparams):
        self.hparams_dict = hparams_dict
        self.top_hparams = top_hparams

    def create_hparams(self):
        """hparams_dict are all different Hparams defined, top_hparams define """
        parser = argparse.ArgumentParser(description='Music generator configuration')
        self.hparams_as_parser(parser)
        new_params = parser.parse_args().__dict__
        hparams = self.update_hparams_from_parser(new_params)
        return hparams

    def hparams_as_parser(self, parser):
        self.top_hparams.iter(lambda key, value:
                              fst(value, parser.add_argument(f"--{key}", dest=key, default=None,
                                                             type=self.type_dict.get(type(value), str),
                                                             help=self.help_dictionary.get(key))),
                              )

    def update_hparams_from_parser(self, new_hparam):
        return self.top_hparams.iter(lambda key, value: self.preprocess_value(new_hparam[key], value), modify=True)

    def preprocess_value(self, new_val, old_val):
        """parse value given by user"""
        if new_val is None:  # argument not given by user
            return old_val
        if isinstance(old_val, Hparams) and isinstance(new_val, str):
            return self.hparams_dict[new_val]
        elif isinstance(old_val, int) and isinstance(new_val, str):
            return int(new_val)
        elif (isinstance(old_val, tuple) or isinstance(old_val, list)) and isinstance(new_val, str):
            vals = new_val[1:-1].replace(" ", "")
            vals = vals.split(",") if "," in vals else [] if vals == '' else [vals]
            vals = [self.preprocess_value(nv, ov) for ov, nv in zip(old_val, vals)]
            return vals
        else:
            return new_val


class Hparams(SimpleNamespace, Mapping):
    def iter(self, fun, modify=False):
        def monad(value, key=None):
            value = fun(key, value) if key is not None else value
            if isinstance(value, Hparams):  # parse substructure
                for new_key, new_val in value.items():
                    x = monad(new_val, f"{key}.{new_key}" if key is not None else new_key)
                    if modify:
                        value[new_key] = x
            elif isinstance(value, list):
                for i, new_val in enumerate(value):
                    x = monad(new_val, f"{key}[{i}]")
                    if modify:
                        value[i] = x
            return value
        return monad(self)

    def items(self):
        return self.__dict__.items()

    def __iter__(self):
        return self.__dict__.__iter__()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        return self.__dict__.__setitem__(key, value)
