import pathlib

import numpy as np
import os

import json
import torch
from torch.utils.data import Dataset
from os import listdir
from os.path import join
from tqdm import tqdm
from utils.misc import load, save
import torchaudio
from functools import reduce
from itertools import repeat
from utils.misc import flatten
import librosa
from typing import NamedTuple
from torch.utils.data import random_split, Subset
import warnings


def get_duration(file):
    return librosa.get_duration(filename=file)


def flatten_dir(dirs):
    return flatten([[join(dir, f) for f in listdir(dir)] for dir in dirs if os.path.isdir(dir)])


class Chunk(NamedTuple):
    file: str
    start: int  # in seconds
    end: int  # in seconds
    file_length: float  # in seconds
    sr: int
    sample_len: int
    channel_level_bias: float

    def read(self) -> np.ndarray:
        # do not quantise tracks into regular chunks, add another offset
        # but also bias towards good beginnings, so do not offset first chunk
        sample_time = self.sample_len / self.sr
        start_offset = torch.rand(1) * sample_time
        start = self.start + start_offset
        end = self.end + start_offset
        duration = (end - start) * 1.1  # add a little bit of data while reading, then forget about it
        if self.file_length - end < 0.1 * sample_time:
            duration = None  # read whole file if end is too close

        sound = self.load_file(start, duration)
        sound = torch.from_numpy(sound[:, :self.sample_len])  # trim sample to at most expected size (can be smaller)
        sound = self.reduce_stereo(sound)
        sound = self.pad_sound(sound)
        return sound.unsqueeze(1)

    def load_file(self, start, duration):
        with warnings.catch_warnings(record=True) as w:
            sound, file_sr = librosa.load(self.file, offset=start, duration=duration, mono=False, sr=self.sr)
        assert file_sr == self.sr  # ensure correct sampling rate
        if len(sound.shape) == 1:
            sound = sound.reshape(1, -1)
        return sound

    def reduce_stereo(self, sound):
        if sound.shape[0] == 1:
            return sound[0]
        elif sound.shape[0] != 2:
            raise Exception(f"Loaded wave with unexpected shape {sound.shape}")
        lvl_bias = torch.rand(1) * self.channel_level_bias
        lvl_bias = torch.cat([0.5 + lvl_bias, 0.5 - lvl_bias])
        sound = torch.matmul(lvl_bias, sound)
        return sound

    def pad_sound(self, sound):
        if sound.shape[0] < self.sample_len:
            padding = torch.zeros((self.sample_len - sound.shape[0],), dtype=sound.dtype)
            sound = torch.cat([sound, padding])
        return sound

    def __str__(self):
        return f"{self.file} from {self.start}s to {self.end}s"

    def __repr__(self):
        return str(self)


class MusicDataset(Dataset):
    def __init__(self,  sound_dirs, sample_len, depth=1, sr=22050, transform=None, min_length=44100,
                 cache_name="file_lengths.pickle", channel_level_bias=0.25):
        self.sound_dirs = sound_dirs if isinstance(sound_dirs, list) else [sound_dirs]
        self.sample_len = sample_len
        self.channel_level_bias = channel_level_bias
        self.min_length = min_length
        self.sr = sr
        self.depth = depth
        self.transform = transform
        self.legal_suffix = [".wav", ".mp3"]
        self.files = self.calculate_files()
        self.sizes, self.dataset_size = self.calculate_lengths(cache_name)
        self.chunks = self.calculate_chunks()

    def calculate_files(self):
        files = reduce(lambda x, f: f(x), repeat(flatten_dir, self.depth), self.sound_dirs)
        files = [f for f in files if f[-4:] in self.legal_suffix]
        return files

    def calculate_lengths(self, cache_path):
        path = pathlib.Path(self.sound_dirs[0]) / cache_path if cache_path is not None else None
        if path is not None and os.path.exists(path):
            print(f"File Lengths loaded from {path}")
            files, sizes, dataset_size = load(str(path))
            if self.files != files or True:  # maybe order is different
                print("Fixing order for cached files")
                #assert len(files) == len(self.files)
                assignment = {file: size for file, size in zip(files, sizes)}
                sizes = [assignment.get(file, 0) for file in self.files]
                old_size = dataset_size
                dataset_size = sum(sizes)
                print(f"Order fixed, {(1-dataset_size/old_size)*100:.5}% information lost")

            return sizes, dataset_size
        sizes = [get_duration(f) for f in tqdm(self.files, desc="Calculating lengths for dataloaders", smoothing=0)]
        dataset_size = sum(sizes)
        if cache_path is not None:
            save((self.files, sizes, dataset_size), str(path))
        return sizes, dataset_size

    def calculate_chunks(self):
        return flatten(self.get_chunks(file, size) for file, size in
                       zip(tqdm(self.files, desc="Dividing dataset into chunks"), self.sizes))

    def get_chunks(self, file: str, size: float) -> [Chunk]:
        if size <= self.min_length:
            return []
        len = self.sample_len / self.sr
        return [Chunk(file, i * len, (i + 1) * len, size, self.sr, self.sample_len, self.channel_level_bias)
                for i in range(int(size / len))]

    def __getitem__(self, idx):
        sound = self.chunks[idx].read()
        return self.transform(sound) if self.transform else sound

    def find_files_for_a_split(self, test_perc):
        wanted_size = self.dataset_size * test_perc
        current_size = 0
        order = torch.randperm(len(self.files), generator=torch.Generator().manual_seed(42))
        file_id = 0
        while current_size + self.sizes[order[file_id]] < wanted_size:
            current_size += self.sizes[order[file_id]]
            yield self.files[order[file_id]]
            file_id += 1
        print(f"test size = {current_size}s, train size = {self.dataset_size - current_size}s.")

    def split_into_two(self, test_perc=0.1, verbeose=False) -> (Dataset, Dataset):
        test_files = set(self.find_files_for_a_split(test_perc))
        if verbeose:
            print(f"Files used for test:\n{','.join(test_files)}")
        train_indices, test_indices = [], []
        for i, chunk in enumerate(self.chunks):
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

    def __len__(self):
        return len(self.chunks)

    def __str__(self):
        return f"{self.dataset_size/60/60:.2f}h music dataset with {len(self)} " \
               f"chunks of size {(self.sample_len / self.sr):.2f}s, located at {self.sound_dirs}"

    def __repr__(self):
        return str(self)
