import audiofile
import librosa
import soundfile
from pathlib import Path


def get_duration(file, use_audiofile):
    return audiofile.duration(file) if use_audiofile else librosa.get_duration(filename=file)


# may return wrong sampling rate! check that
def load_file(filename, start=0, duration=None, sr=None, use_audiofile=False):
    return audiofile.read(filename, offset=start, duration=duration) if use_audiofile else \
        librosa.load(filename, offset=start, duration=duration, mono=False, sr=sr)


def save_file(data, filename, sr):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(data, filename, samplerate=sr)