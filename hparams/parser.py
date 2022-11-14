import argparse
from types import SimpleNamespace
from collections.abc import Mapping
import logging

from utils.misc import fst


class HparamsParser:
    help_dictionary = {
        "model": "type of model: 'prior' or 'compressor' pr 'upsampler'"
    }
    type_dict = {int: int, float: float}

    def __init__(self, hparams_registry):
        self.hparams_registry = hparams_registry
        self.default_hparams = self.hparams_registry["default"]

    def create_hparams(self):
        """hparams_dict are all different Hparams defined, top_hparams define """
        parser = argparse.ArgumentParser(description='Music generator configuration')
        self.hparams_as_parser(parser)
        new_params = parser.parse_args().__dict__
        hparams = self.update_hparams_from_parser(new_params)
        return hparams

    def hparams_as_parser(self, parser):
        self.default_hparams.iter(lambda key, value:
                                  fst(value, parser.add_argument(f"--{key or 'config'}", default=None,
                                                                 type=self.type_dict.get(type(value), str),
                                                                 help=self.help_dictionary.get(key))),
                                  )

    def update_hparams_from_parser(self, new_hparam):
        return self.default_hparams.iter(lambda key, value:
                                         self.preprocess_value(new_hparam.get(key or "config", None), value), modify=True)

    def preprocess_value(self, new_val, old_val):
        """parse value given by user"""
        if new_val is None:  # argument not given by user
            return old_val
        if isinstance(old_val, Hparams) and isinstance(new_val, str):
            return self.hparams_registry[new_val]
        elif isinstance(old_val, int) and isinstance(new_val, str):
            return int(new_val)
        elif (isinstance(old_val, tuple) or isinstance(old_val, list)) and isinstance(new_val, str):
            vals = new_val[1:-1].replace(" ", "")
            vals = vals.split(",") if "," in vals else [] if vals == '' else [vals]
            if len(old_val) == 0:
                logging.warning(f"Unknown type for list value {new_val}, because previous {old_val} is empty")
                ov = None
            else:
                ov = old_val[0]
            vals = [self.preprocess_value(nv, ov) for nv in vals]
            return vals
        else:
            return new_val


class Hparams(SimpleNamespace, Mapping):
    def iter(self, fun, modify=False):
        def monad(value, key=None):
            value = fun(key, value)
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

    def update(self, **args):
        d = self.__dict__.copy()
        d.update(args)
        return Hparams(**d)
