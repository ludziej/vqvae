import logging
import multiprocessing as mp
import queue
import warnings
from typing import NamedTuple, List, Any

import librosa
import numpy
import numpy as np
import torch

from data_processing.tools import load_file
from utils.misc import time_run
from data_processing.normalization import rms_normalize


class TimeConditioning(NamedTuple):
    start: float  # seconds
    end: float
    total: float


class ContextConditioning(NamedTuple):
    artist: int  # class
    genres: List[int]
    listens: int


class DatasetConfig(NamedTuple):
    channel_level_bias: float
    use_audiofile: bool  # if False uses librosa
    another_thread: bool
    sample_len: int
    timeout: int
    sr: int
    logger: Any
    rms_normalize_sound: bool
    rms_normalize_level: float
    #artist_total: int
    genre_total: int
    #artist_names: List[str]
    #genre_names: List[str]


class Track(NamedTuple):
    config: DatasetConfig
    filename: str
    file_length: float  # in seconds
    artist: int
    genres: List[int]
    listens: int

    def get_context_cond(self):
        return ContextConditioning(artist=self.artist, genres=self.genres, listens=self.listens)

    def __str__(self):
        return f"{self.filename}[total {self.file_length:.1f}s]"


class Chunk(NamedTuple):
    track: Track
    start: int  # in seconds
    end: int  # in seconds

    def read(self) -> np.ndarray:
        # do not quantise tracks into regular chunks, add another offset
        # but also bias towards good beginnings, so do not offset first chunk
        sample_time = self.track.config.sample_len / self.track.config.sr
        start_offset = torch.rand(1) * sample_time
        start = self.start + start_offset
        end = self.end + start_offset
        duration = (end - start) * 1.1  # add a little bit of data while reading, then forget about it
        if self.track.file_length - end < 0.1 * sample_time:
            duration = None  # read whole file if end is too close

        sound = self.load_file(start, duration)
        sound = torch.from_numpy(sound[:, :self.track.config.sample_len])  # trim sample to at most expected size (can be smaller)
        sound = self.reduce_stereo(sound)
        sound = self.pad_sound(sound)
        if self.track.config.rms_normalize_sound:
            sound = rms_normalize(sound, self.track.config.rms_normalize_level)
        return sound.unsqueeze(1)

    def resample(self, sound, from_sr):
        return librosa.resample(sound, orig_sr=from_sr, target_sr=self.track.config.sr)

    # multiprocessing stuff, because library is breaking sometimes

    def _load_file_into_queue(self, start, duration, use_audiofile, data_queue):
        data_queue.put(self._load_file(start, duration, use_audiofile))

    def _load_file_another_thread(self, start, duration):
        data_queue = mp.Queue()
        task = mp.Process(target=self._load_file_into_queue,
                          args=(start, duration, self.track.config.use_audiofile, data_queue))
        task.start()
        output = None
        trycount = 1
        timeout = self.track.config.timeout
        while output is None:
            try:
                output = data_queue.get(block=True, timeout=timeout)
            except queue.Empty:
                task.terminate()
                task = mp.Process(target=self._load_file_into_queue,
                                  args=(start, duration, self.track.config.use_audiofile, data_queue))
                task.start()
                trycount += 1
                timeout *= 2
                self.track.config.logger.info(f"Hanged {str(self)} read, running again, try nr {trycount}, timeout={timeout}")
        data_queue.close()
        task.join()
        return output

    def _load_file(self, start, duration, use_audiofile):
        try:
            return load_file(self.track.filename, start.item(), duration.item() if duration is not None else None,
                      use_audiofile=use_audiofile, sr=self.track.config.sr)
        except Exception as e:
            self.track.config.logger.warning(f"Exception {e} in {str(self)} config = {self.track.config}")
            return numpy.zeros((1, 1), float)

    def get_time_cond(self):
        return TimeConditioning(self.start, self.end, self.track.file_length)

    def get_context_cond(self):
        return self.track.get_context_cond()

    def load_file(self, start, duration):
        run_lambda = (lambda: self._load_file_another_thread(start, duration)) if self.track.config.another_thread \
            else (lambda: self._load_file(start, duration, self.track.config.use_audiofile))
        if logging.DEBUG >= self.track.config.logger.level:
            time, (sound, file_sr) = time_run(run_lambda)
            self.track.config.logger.debug(f"Reading from '{str(self)}' took {time:.2f} seconds")
        else:
            with warnings.catch_warnings(record=True) as w:
                time, (sound, file_sr) = 0, run_lambda()
                for warn in w:
                    self.track.config.logger.info(f"{warn.category} : {warn.message}")
        if file_sr != self.track.config.sr and len(sound.shape) >= 2 and sound.shape[1] > 50:
            time, sound = time_run(lambda: self.resample(sound, file_sr))
            self.track.config.logger.debug(f"Resampling from '{str(self)}' because of wrong sr={file_sr}, took {time:.2f}s")
        if len(sound.shape) == 1:
            sound = sound.reshape(1, -1)
        return sound

    def reduce_stereo(self, sound):
        if sound.shape[0] == 1:
            return sound[0]
        elif sound.shape[0] != 2:
            raise Exception(f"Loaded wave with unexpected shape {sound.shape}")
        lvl_bias = torch.rand(1) * self.track.config.channel_level_bias
        lvl_bias = torch.cat([0.5 + lvl_bias, 0.5 - lvl_bias])
        sound = torch.matmul(lvl_bias, sound)
        return sound

    def pad_sound(self, sound):
        if sound.shape[0] < self.track.config.sample_len:
            padding = torch.zeros((self.track.config.sample_len - sound.shape[0],), dtype=sound.dtype)
            sound = torch.cat([sound, padding])
        return sound

    def __str__(self):
        return f"{str(self.track)} from {self.start:.1f}s to {self.end:.1f}s"

    def __repr__(self):
        return str(self)