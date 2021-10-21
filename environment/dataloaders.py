import numpy as np
import os

import torch
from torch.utils.data import Dataset
from os import listdir
from os.path import join
import torchaudio
from functools import reduce
from itertools import repeat
from utils.misc import flatten


class WaveDataset(Dataset):
    def __init__(self,  sound_dirs, sample_len, depth=1, sr=22050, transform=None):
        self.sound_dirs = sound_dirs if isinstance(sound_dirs, list) else [sound_dirs]
        self.sample_len = sample_len
        self.sr = sr
        self.transform = transform
        self.files = reduce(lambda x, f: f(x), repeat(self.get_sons, depth), self.sound_dirs)
        self.files = [f for f in self.files if f[-4:] == ".wav"]
        self.randgen = np.random.default_rng(os.getpid())

    def get_sons(self, dirs):
        return flatten([[join(dir, f) for f in listdir(dir)] for dir in dirs if os.path.isdir(dir)])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        whole_file, file_sr = torchaudio.load(self.files[idx])
        assert file_sr == self.sr
        # reduce stereo
        whole_file = (whole_file[0:1, :] + whole_file[1:2, :])/2 if whole_file.shape[0] >= 2 else whole_file

        randomize_start = self.randgen.integers(low=0, high=whole_file.shape[1] - 1, size=1)[0]
        trimmed_sound = whole_file[:, randomize_start:randomize_start + self.sample_len]
        if trimmed_sound.shape[1] < self.sample_len:
            padding = torch.zeros((1, self.sample_len - trimmed_sound.shape[1]), dtype=trimmed_sound.dtype)
            trimmed_sound = torch.cat([trimmed_sound, padding], axis=1)
        if self.transform:
            trimmed_sound = self.transform(trimmed_sound)
        return trimmed_sound.T
