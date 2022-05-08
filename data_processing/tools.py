import audiofile
import librosa
import soundfile


def get_duration(file, use_audiofile):
    return audiofile.duration(file) if use_audiofile else librosa.get_duration(filename=file)


# may return wrong sampling rate! check that
def load_file(filename, start=0, duration=None, sr=None, use_audiofile=False):
    return audiofile.read(filename, offset=start, duration=duration) if use_audiofile else \
        librosa.load(filename, offset=start, duration=duration, mono=False, sr=sr)


def save_file(filename, sr):
    soundfile.write(filename, samplerate=sr)