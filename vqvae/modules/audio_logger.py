import os

import torch
import torch.nn as nn
from nnAudio.Spectrogram import STFT
import librosa.display
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from neptune.types import File
import uuid
from data_processing.tools import save_file, load_file
import tempfile


class AudioLogger(nn.Module):
    def __init__(self, sr, model, use_log_grads=False, use_weights_logging=False,
                 logger_type="tensorboard"):
        super(AudioLogger, self).__init__()
        self.use_log_grads = use_log_grads
        self.logger_type = logger_type
        self.model = [model]
        self.sr = sr
        self.use_weights_logging = use_weights_logging
        self.elogger = lambda: self.model[0].logger.experiment
        self.my_log = lambda *x, **xx: self.model[0].log(*x, **xx)
        self.spec = STFT(n_fft=512, center=True, hop_length=128, sr=self.sr, verbose=False)
        self.log_nr = {"val_": 0, "": 0, "test_": 0}

    @torch.no_grad()
    def log_grads(self):
        if not self.use_log_grads:
            return
        metrics = {}
        for name, value in self.model[0].named_parameters():
            if value.grad is not None:
                norm = torch.linalg.norm(value.grad)
                metrics[f"grad_norm/{name}"] = norm
        metrics[f"grad_norm_total"] = torch.linalg.norm(torch.tensor(list(metrics.values())))
        self.log_add_metrics(metrics, prog_bar=False)

    def next_log_nr(self, prefix):
        nr = self.log_nr.get(prefix, 0)
        self.log_nr[prefix] = nr + 1

    def log_weights_norm(self):
        metrics = {}
        for name, value in self.model[0].named_parameters():
            if value.requires_grad:
                norm = torch.linalg.norm(value)
                metrics[f"weight_norm/{name}"] = norm
        return metrics

    def log_add_metrics(self, metrics, prefix="", prog_bar=True):
        for k, v in metrics.items():
            k = prefix + k
            v = v.item() if isinstance(v, torch.Tensor) else v
            if self.logger_type == "tensorboard":
                self.my_log(k, v, prog_bar=prog_bar, rank_zero_only=True)
            elif self.logger_type == "neptune":
                self.elogger()[k].append(v)
            else:
                raise Exception("Not Implemented")

    def log_metrics(self, metrics, prefix=""):
        self.next_log_nr(prefix)
        if self.use_weights_logging:
            metrics.update(**self.log_weights_norm())
        self.log_add_metrics(metrics, prefix)

    def log_sound(self, name, sound, nr, desc=""):
        if self.logger_type == "tensorboard":
            self.elogger().add_audio(name, sound, nr, self.sr)
        elif self.logger_type == "neptune":
            filename = f"{tempfile.gettempdir()}/{uuid.uuid4()}.wav"
            save_file(sound.squeeze(0), filename, self.sr)
            self.elogger()[name].upload(filename)
            #os.remove(filename)
        else:
            raise Exception("Not Implemented")

    def log_image(self, name, image, nr, desc=""):
        if self.logger_type == "tensorboard":
            self.elogger().add_image(name, image, nr)
        elif self.logger_type == "neptune":
            self.elogger()[name].append(File.as_image(np.transpose(image/255, (1, 2, 0))))
        else:
            raise Exception("Not Implemented")

    def log_sounds(self, batch, name, prefix):
        for i, xin in enumerate(batch):
            self.log_sound(prefix + name(i), xin, self.log_nr.get(prefix, 0))

    def get_fft(self, sound):
        return self.spec(sound).permute(0, 3, 1, 2)

    def get_plot(self, fft):
        ampl = (fft[0]**2 + fft[1]**2)**(1/2)

        fig, ax = plt.subplots()
        data = ampl.detach().cpu().numpy()
        dbb = librosa.amplitude_to_db(data, ref=np.max)
        img = librosa.display.specshow(dbb, x_axis='time', y_axis="linear", ax=ax, hop_length=128, sr=self.sr)
        fig.colorbar(img, ax=ax)
        fig.tight_layout()
        fig.canvas.draw()
        X = np.array(fig.canvas.renderer.buffer_rgba())
        fig.clf()
        plt.close()
        return X

    def plot_spec_as(self, sounds, name, prefix):
        nr = self.log_nr.get(prefix, 0)
        ffts = self.get_fft(sounds)
        for i, fft in enumerate(ffts):
            image = self.get_plot(fft)
            self.log_image(prefix + name(i), np.transpose(image, (2, 0, 1)), nr)

    def plot_spectrorams(self, batch, batch_outs, prefix):
        self.plot_spec_as(batch, f"in", prefix)
        for level, lvl_outs in enumerate(batch_outs):
            self.plot_spec_as(lvl_outs, f"out_lvl_{level}", prefix)
