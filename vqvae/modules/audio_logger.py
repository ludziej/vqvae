import torch
import torch.nn as nn
import torch as t
from nnAudio.Spectrogram import CQT2010v2, STFT
import librosa
import librosa.display
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


class AudioLogger(nn.Module):
    def __init__(self, sr):
        super(AudioLogger, self).__init__()
        self.sr = sr
        self.spec = STFT(n_fft=512, center=True, hop_length=128, sr=self.sr, verbose=False)

    def get_fft(self, sound):
        return self.spec(sound).permute(0, 3, 1, 2)

    def get_plot(self, fft):
        ampl = (fft[0]**2 + fft[1]**2)**(1/2)

        fig, ax = plt.subplots()
        data = ampl.detach().cpu().numpy()
        dbb = librosa.amplitude_to_db(data, ref=np.max)
        img = librosa.display.specshow(dbb, x_axis='time', y_axis="linear", ax=ax,
                                       hop_length=128,
                                       sr=self.sr)
        fig.colorbar(img, ax=ax)
        fig.tight_layout()
        fig.canvas.draw()
        X = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        return X

    def plot_spec_as(self, sounds, name, nr):
        ffts = self.get_fft(sounds)
        for i, fft in enumerate(ffts):
            image = self.get_plot(fft)
            self.logger.experiment.add_image(f"spec_{i}/{name}", np.transpose(image, (2, 0, 1)), nr)

    def plot_spectrorams(self, batch, batch_outs, nr):
        self.plot_spec_as(batch, f"in", nr)
        for level, lvl_outs in enumerate(batch_outs):
            self.plot_spec_as(lvl_outs, f"out_lvl_{level}", nr)
