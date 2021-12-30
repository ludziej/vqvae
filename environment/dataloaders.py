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
import librosa
from typing import NamedTuple
from torch.utils.data import random_split, Subset


def get_duration(file):
    return librosa.get_duration(filename=file)


def flatten_dir(dirs):
    return flatten([[join(dir, f) for f in listdir(dir)] for dir in dirs if os.path.isdir(dir)])


class Chunk(NamedTuple):
    file: str
    start: int
    end: int
    file_length: int
    sr: int
    sample_len: int
    channel_level_var: float

    def read(self) -> np.ndarray:
        # do not quantise tracks into regular chunks, add another offset
        # but also bias towards good beginnings, so do not offset first chunk
        start_offset = self.randgen.integers(low=0, high=self.sample_len - 5, size=1)[0] if self.start != 0 else 0
        start = self.start + start_offset
        duration = (self.end - start_offset) * 1.1  # add a little bit of data while reading, then forget about it
        if self.file_length - (duration + start) < 0.1 * self.sample_len:
            duration = None  # read whole file if end is too close
        else:
            duration *= self.sr
        start *= self.sr
        sound, file_sr = librosa.load(self.file, offset=start, duration=duration)
        assert file_sr == self.sr  # ensure correct sampling rate
        sound = sound[:, :self.sample_len]  # trim sample to at most expected size (can be smaller)
        sound = self.reduce_stereo(sound)
        sound = self.pad_sound(sound)
        return sound

    def reduce_stereo(self, sound):
        lvl = torch.rand(1) * self.channel_level_var
        lvl = torch.cat([0.5 + lvl, 0.5 - lvl])
        sound = torch.dot(lvl, sound)
        return sound
        #whole_file = (whole_file[0:1, :] + whole_file[1:2, :])/2 if whole_file.shape[0] >= 2 else whole_file
        #sound = torch.mean(sound, dim=0, keepdim=True)

    def pad_sound(self, sound):
        if sound.shape[1] < self.sample_len:
            padding = torch.zeros((1, self.sample_len - sound.shape[1]), dtype=sound.dtype)
            sound = torch.cat([sound, padding], axis=1)
        return sound

    def __str__(self):
        return f"{self.file} from {self.start} to {self.end}"

    def __repr__(self):
        return str(self)


class WaveDataset(Dataset):
    def __init__(self,  sound_dirs, sample_len, depth=1, sr=22050, transform=None, channel_level_var=0.25):
        self.sound_dirs = sound_dirs if isinstance(sound_dirs, list) else [sound_dirs]
        self.sample_len = sample_len
        self.channel_level_var = channel_level_var
        self.sr = sr
        self.transform = transform
        self.files = reduce(lambda x, f: f(x), repeat(flatten_dir, depth), self.sound_dirs)
        self.files = [f for f in self.files if f[-4:] == ".wav"]
        self.sizes = [get_duration(f) for f in self.files]
        self.dataset_size = sum(self.sizes)
        self.chunks = flatten(self.get_chunks(file, size) for file, size in zip(self.files, self.sizes))

    def __len__(self):
        return len(self.chunks)

    def get_chunks(self, file: str, size: int) -> [Chunk]:
        len = self.sample_len
        return [Chunk(file, i * len, (i + 1) * len, size, self.sr, len, self.channel_level_var)
                for i in range(size / len)]

    def __getitem__(self, idx):
        sound = self.chunks[idx].read()
        return (self.transform(sound) if self.transform else sound).T

    def find_files_for_a_split(self, test_perc):
        wanted_size = self.dataset_size * test_perc
        current_size = 0
        file_id = 0
        while current_size + self.sizes[file_id] < wanted_size:
            current_size += self.sizes[file_id]
            yield self.files[file_id]
            file_id += 1
        print(f"test size = {current_size}s, train size = {self.dataset_size - current_size}s.")

    def split_into_two(self, test_perc=0.1) -> (Dataset, Dataset):
        test_files = set(self.find_files_for_a_split(test_perc))
        print(f"Files used for test:\n{','.join(self.test_files)}")
        train_indices, test_indices = [], []
        for i, chunk in self.enumerate(self.chunks):
            (test_indices if chunk.file in test_files else train_indices).append(i)
        train_data = Subset(self, train_indices)
        test_data = Subset(self, test_indices)
        return train_data, test_data

    # this split may be leaking
    def split_into_two_by_chunks(self, test_perc=0.1) -> (Dataset, Dataset):
        train_size = int((1 - test_perc) * len(self))
        train_data, test_data = random_split(self, [train_size, len(self) - train_size],
                                             generator=torch.Generator().manual_seed(42))
        print(f"Chunks used for test:\n{','.join([self.chunks[i] for i in test_data.indices])}")
        return train_data, test_data
