import os
import pickle
from typing import Callable
from time import time
import logging
import json


def time_run(fun):
    t = time()
    val = fun()
    return time() - t, val


def save(obj, filename: str):
    with open(filename, "wb") as output_file:
        pickle.dump(obj, output_file)


def load(filename: str):
    with open(filename, "rb") as input_file:
        return pickle.load(input_file)


def load_json(filename: str):
    with open(filename) as json_file:
        return json.load(json_file)


def lazy_compute_pickle(fun: Callable[[], object], filename: str) -> object:
    if os.path.exists(filename):
        logging.info(f"Reading cached values from {filename}")
        return load(filename)
    obj = fun()
    save(obj, filename)
    return obj


def flatten(t):
    return [item for sublist in t for item in sublist]


def reverse_mapper(mapper):
    inv_map = {v: k for k, v in mapper.items()}
    return [inv_map[i] for i in range(len(inv_map))]


def fst(x, y):
    return x