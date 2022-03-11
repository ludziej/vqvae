import os
import pickle
from typing import Callable
from time import time


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


def lazy_compute_pickle(fun: Callable[[], object], filename: str) -> object:
    if os.path.exists(filename):
        return load(filename)
    obj = fun()
    save(obj, filename)
    return obj


def flatten(t):
    return [item for sublist in t for item in sublist]


def fst(x, y):
    return x