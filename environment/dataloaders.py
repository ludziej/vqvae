import logging
import pathlib
import time

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
from utils.misc import time_run
import audiofile
import multiprocessing as mp
import queue
import pathlib


def get_duration(file, use_audiofile):
    return audiofile.duration(file) if use_audiofile else librosa.get_duration(filename=file)


def flatten_dir(dirs):
    return flatten([[join(dir, f) for f in listdir(dir)] for dir in dirs if os.path.isdir(dir)])


class ChunkConfig(NamedTuple):
    channel_level_bias: float
    use_audiofile: bool  # if False uses librosa
    another_thread: bool
    timeout: int


class Chunk(NamedTuple):
    file: str
    start: int  # in seconds
    end: int  # in seconds
    file_length: float  # in seconds
    sr: int
    sample_len: int
    chunk_config: ChunkConfig

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

    def resample(self, sound, from_sr):
        return librosa.resample(sound, orig_sr=from_sr, target_sr=self.sr)

    # multiprocessing stuff, because library is breaking sometimes

    def _load_file_into_queue(self, start, duration, use_audiofile, data_queue):
        data_queue.put(self._load_file(start, duration, use_audiofile))

    def _load_file_another_thread(self, start, duration):
        data_queue = mp.Queue()
        task = mp.Process(target=self._load_file_into_queue,
                          args=(start, duration, self.chunk_config.use_audiofile, data_queue))
        task.start()
        output = None
        trycount = 1
        timeout = self.chunk_config.timeout
        while output is None:
            try:
                output = data_queue.get(block=True, timeout=timeout)
            except queue.Empty:
                task.terminate()
                task = mp.Process(target=self._load_file_into_queue,
                                  args=(start, duration, self.chunk_config.use_audiofile, data_queue))
                task.start()
                trycount += 1
                timeout *= 2
                logging.info(f"Hanged {str(self)} read, running again, try nr {trycount}, timeout={timeout}")
        data_queue.close()
        task.join()
        return output

    def _load_file(self, start, duration, use_audiofile):
        if use_audiofile:
            return audiofile.read(self.file, offset=start.item(),
                                  duration=duration.item() if duration is not None else None)
        return librosa.load(self.file, offset=start, duration=duration, mono=False, sr=self.sr)

    def load_file(self, start, duration):
        run_lambda = (lambda: self._load_file_another_thread(start, duration)) if self.chunk_config.another_thread \
            else (lambda: self._load_file(start, duration, self.chunk_config.use_audiofile))
        if logging.DEBUG >= logging.root.level:
            time, (sound, file_sr) = time_run(run_lambda)
            logging.debug(f"Reading from '{str(self)}' took {time:.2f} seconds")
        else:
            with warnings.catch_warnings(record=True) as w:
                time, (sound, file_sr) = 0, run_lambda()
                for warn in w:
                    logging.debug(f"{warn.category} : {warn.message}")
        if file_sr != self.sr and len(sound.shape) >= 2 and sound.shape[1] > 50:
            time, sound = time_run(lambda: self.resample(sound, file_sr))
            logging.debug(f"Resampling from '{str(self)}' because of wrong sr={file_sr}, took {time:.2f}s")
        if len(sound.shape) == 1:
            sound = sound.reshape(1, -1)
        return sound

    def reduce_stereo(self, sound):
        if sound.shape[0] == 1:
            return sound[0]
        elif sound.shape[0] != 2:
            raise Exception(f"Loaded wave with unexpected shape {sound.shape}")
        lvl_bias = torch.rand(1) * self.chunk_config.channel_level_bias
        lvl_bias = torch.cat([0.5 + lvl_bias, 0.5 - lvl_bias])
        sound = torch.matmul(lvl_bias, sound)
        return sound

    def pad_sound(self, sound):
        if sound.shape[0] < self.sample_len:
            padding = torch.zeros((self.sample_len - sound.shape[0],), dtype=sound.dtype)
            sound = torch.cat([sound, padding])
        return sound

    def __str__(self):
        return f"{self.file}[total {self.file_length:.1f}s] from {self.start:.1f}s to {self.end:.1f}s"

    def __repr__(self):
        return str(self)


class MusicDataset(Dataset):
    def __init__(self,  sound_dirs, sample_len,  sr=22050, transform=None, min_length=1, timeout=1,
                 cache_name="file_lengths.pickle", channel_level_bias=0.25, use_audiofile=False, another_thread=False):
        self.sound_dirs = sound_dirs if isinstance(sound_dirs, list) else [sound_dirs]
        self.sample_len = sample_len
        self.channel_level_bias = channel_level_bias
        self.min_length = min_length
        self.sr = sr
        self.transform = transform
        self.use_audiofile = use_audiofile
        self.legal_suffix = [".wav", ".mp3" ".ape", ".flac"]
        self.files = flatten(self.get_music_in(pathlib.Path(x)) for x in self.sound_dirs)
        self.cache_dir = pathlib.Path(self.sound_dirs[0])
        self.cache_path = self.cache_dir / cache_name if cache_name is not None else None
        self.sizes, self.dataset_size = self.calculate_lengths()
        self.chunk_config = ChunkConfig(channel_level_bias, use_audiofile, another_thread, timeout)
        self.chunks = self.calculate_chunks()

    def get_music_in(self, file):
        if os.path.isdir(file):
            return flatten(self.get_music_in(file / x) for x in listdir(file))
        elif any(str(file).endswith(suf) for suf in self.legal_suffix):
            return [str(file)]
        return []

    def calculate_lengths(self):
        if self.cache_path is not None and os.path.exists(self.cache_path):
            logging.info(f"File Lengths loaded from {self.cache_path}")
            files, sizes, dataset_size = load(str(self.cache_path))
            if self.files != files:  # maybe order is different
                logging.info(f"Fixing order for cached files, prev {len(files)} now {len(self.files)} files, "
                             f"\n then {files[:1]} now {self.files[:1]}")
                #assert len(files) == len(self.files)
                assignment = {os.path.basename(file): size for file, size in zip(files, sizes)}
                sizes = [assignment.get(os.path.basename(file), 0) for file in self.files]
                old_size = dataset_size
                dataset_size = sum(sizes)
                logging.info(f"Order fixed, {(1-dataset_size/old_size)*100:.5}% information lost")

            return sizes, dataset_size
        sizes = [get_duration(f, self.use_audiofile) for f in
                 tqdm(self.files, desc="Calculating lengths for dataloaders", smoothing=0)]
        dataset_size = sum(sizes)
        if self.cache_path is not None:
            save((self.files, sizes, dataset_size), str(self.cache_path))
        return sizes, dataset_size

    def calculate_chunks(self):
        return flatten(self.get_chunks(file, size) for file, size in
                       zip(tqdm(self.files, desc="Dividing dataset into chunks"), self.sizes))

    def get_chunks(self, file: str, size: float) -> [Chunk]:
        if size <= self.min_length:
            return []
        len = self.sample_len / self.sr
        return [Chunk(file, i * len, (i + 1) * len, size, self.sr, self.sample_len, self.chunk_config)
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
        logging.info(f"test size = {current_size}s, train size = {self.dataset_size - current_size}s.")

    def split_into_two(self, test_perc=0.1) -> (Dataset, Dataset):
        test_files = set(self.find_files_for_a_split(test_perc))
        logging.log(5, f"Files used for test:\n{','.join(test_files)}")
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
        logging.log(5, f"Chunks used for test:\n{','.join([self.chunks[i] for i in test_data.indices])}")
        return train_data, test_data

    def __len__(self):
        return len(self.chunks)

    def __str__(self):
        return f"{self.dataset_size/60/60:.2f}h music dataset with {len(self)} " \
               f"chunks of size {(self.sample_len / self.sr):.2f}s, located at {self.sound_dirs}"

    def __repr__(self):
        return str(self)
