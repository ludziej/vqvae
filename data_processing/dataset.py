import os

import torch
from torch.utils.data import Dataset
from os import listdir
from tqdm import tqdm

from data_processing.chunks import DatasetConfig, Chunk, Track
from data_processing.tools import get_duration
from utils.misc import load, save, flatten, load_json, reverse_mapper
from torch.utils.data import random_split, Subset
import json
import pandas as pd
import pathlib


class MusicDataset(Dataset):
    def __init__(self,  sound_dirs, sample_len, logger, sr=22050, transform=None, min_length=1, timeout=1,
                 context_cond=False, time_cond=False, cache_name="file_lengths.pickle", channel_level_bias=0.25,
                 use_audiofile=False, another_thread=False, rms_normalize_sound=True, metadata_tracks="tracks.csv",
                 genres_data="genres.csv", rms_normalize_level=-5, banned_genres=()):
        self.sound_dirs = sound_dirs if isinstance(sound_dirs, list) else [sound_dirs]
        self.logger = logger
        self.time_cond = time_cond
        self.context_cond = context_cond
        self.sample_len = sample_len
        self.channel_level_bias = channel_level_bias
        self.min_length = min_length
        self.sr = sr
        self.transform = transform
        self.use_audiofile = use_audiofile
        self.logger = logger
        self.banned_genres = set(banned_genres)
        self.legal_suffix = [".wav", ".mp3", ".ape", ".flac"]
        self.context_file = "context.json"
        self.genres = []
        self.cache_dir = pathlib.Path(self.sound_dirs[0])
        if self.context_cond:
            self.track_metadata, self.genres, self.genre_map, self.genre_reverse_mapper \
                = self.calc_track_metadata(metadata_tracks, genres_data)

        self.empty_context = dict()
        self.files, self.contexts = self.get_files_and_contexts()
        #self.contexts_vec, self.context_names, self.context_totals = self.vectorise_contexts()
        self.cache_path = self.cache_dir / cache_name if cache_name is not None else None
        self.sizes, self.dataset_size = self.calculate_lengths()
        self.chunk_config = DatasetConfig(channel_level_bias, use_audiofile, another_thread, self.sample_len, timeout,
                                          self.sr, logger, rms_normalize_sound, rms_normalize_level,
                                          genre_total=len(self.genres))
        self.chunks = self.calculate_chunks()

    def calc_track_metadata(self, tracks, genres):
        genres = pd.read_csv(self.cache_dir / genres, header=0)
        genre_mapper = {gid: id for id, gid in zip(genres.index, genres.genre_id)}
        genre_legal = {g for g, name in zip(genres.index, genres.title) if name.lower() not in self.banned_genres}
        genre_reverse_mapper = {v: k for k, v in genre_mapper.items()}

        tracks = pd.read_csv(self.cache_dir / tracks, index_col=0, header=[0, 1])
        tracks_data = [(list(map(genre_mapper.get, json.loads(g))), max(l, 1))
                       for g, l in zip(tracks.track.genres_all, tracks.album.listens)]
        tracks_data = {i: (g, l) for i, (g, l) in zip(tracks.index, tracks_data)
                       if all(gi in genre_legal for gi in g)}

        return tracks_data, genres, genre_mapper, genre_reverse_mapper

    def get_files_and_contexts(self):
        data = flatten(self.get_music_in(pathlib.Path(x), self.empty_context) for x in self.sound_dirs)
        if len(data) == 0:
            raise Exception(f"Empty dataset provided at {self.sound_dirs}")
        files, contexts = zip(*data)
        return list(files), list(contexts)

    # recursively gather information through catalog structure
    def get_music_in(self, file, context_dict):
        if os.path.isdir(file):
            if os.path.exists(file / self.context_file):
                context_dict = context_dict.copy()
                context_dict.update(load_json(file / self.context_file))
            return flatten(self.get_music_in(file / x, context_dict) for x in listdir(file))
        elif any(str(file).endswith(suf) for suf in self.legal_suffix):
            return [(str(file), context_dict)]
        return []

    def vectorise_contexts(self):
        # returns: (int-hashed self.contexts, dict with names resolve for each context type,
        #   dict with total length of each context type)
        name_mapper = dict()
        for c in self.contexts:
            for k, v in c.items():
                if k not in name_mapper:
                    name_mapper[k] = {v: 0}
                elif v not in name_mapper[k]:
                    name_mapper[k][v] = len(name_mapper[k])
        context_vec = [{k: name_mapper[k][v] for k, v in context.items()} for context in self.contexts]
        names = {f"{name}_names": reverse_mapper(mapper) for name, mapper in name_mapper.items()}
        lengths = {f"{name}_total": len(v) for name, v in name_mapper.items()}
        return context_vec, names, lengths

    def calculate_lengths(self):
        if self.cache_path is not None and os.path.exists(self.cache_path):
            self.logger.info(f"File Lengths loaded from {self.cache_path}")
            files, sizes, dataset_size = load(str(self.cache_path))
            if self.files != files:  # maybe order is different
                self.logger.info(f"Fixing order for cached files, prev {len(files)} now {len(self.files)} files, "
                             f"\n then {files[:1]} now {self.files[:1]}")
                #assert len(files) == len(self.files)
                assignment = {os.path.basename(file): size for file, size in zip(files, sizes)}
                sizes = [assignment.get(os.path.basename(file), 0) for file in self.files]
                old_size = dataset_size
                dataset_size = sum(sizes)
                self.logger.info(f"Order fixed, {(1-dataset_size/old_size)*100:.5}% information lost")

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
        file_id = int(os.path.basename(file).split('.')[0])
        genres, listens = self.track_metadata.get(file_id, (None, None)) if self.context_cond else ([0], 1)
        if genres is None or len(genres) == 0 or listens == -1:  # banned genre
            return []
        track = Track(self.chunk_config, file, size, 0, genres, listens)
        if size <= self.min_length:
            return []
        length = self.sample_len / self.sr
        return [Chunk(track, i * length, (i + 1) * length) for i in range(int(size / length))]

    def __getitem__(self, idx):
        sound = self.chunks[idx].read()
        sound = self.transform(sound) if self.transform else sound
        additional = dict()
        if self.context_cond:
            additional["context"] = self.chunks[idx].get_context_cond()
        if self.time_cond:
            additional["time"] = self.chunks[idx].get_time_cond()
        return sound, additional

    def find_files_for_a_split(self, test_perc):
        wanted_size = self.dataset_size * test_perc
        current_size = 0
        order = torch.randperm(len(self.files), generator=torch.Generator().manual_seed(42))
        file_id = 0
        while current_size + self.sizes[order[file_id]] < wanted_size:
            current_size += self.sizes[order[file_id]]
            yield self.files[order[file_id]]
            file_id += 1
        self.logger.info(f"test size = {current_size}s, train size = {self.dataset_size - current_size}s.")

    def split_into_two(self, test_perc=0.1) -> (Dataset, Dataset):
        test_files = set(self.find_files_for_a_split(test_perc))
        self.logger.log(5, f"Files used for test:\n{','.join(test_files)}")
        train_indices, test_indices = [], []
        for i, chunk in enumerate(self.chunks):
            (test_indices if chunk.track.filename in test_files else train_indices).append(i)
        train_data = Subset(self, train_indices)
        test_data = Subset(self, test_indices)
        return train_data, test_data

    # this split may be leaking
    def split_into_two_by_chunks(self, test_perc=0.1) -> (Dataset, Dataset):
        train_size = int((1 - test_perc) * len(self))
        train_data, test_data = random_split(self, [train_size, len(self) - train_size],
                                             generator=torch.Generator().manual_seed(42))
        self.logger.log(5, f"Chunks used for test:\n{','.join([self.chunks[i] for i in test_data.indices])}")
        return train_data, test_data

    def __len__(self):
        return len(self.chunks)

    def __str__(self):
        return f"{self.dataset_size/60/60:.2f}h music dataset with {len(self)} " \
               f"chunks of size {(self.sample_len / self.sr):.2f}s, located at {self.sound_dirs}"

    def __repr__(self):
        return str(self)
