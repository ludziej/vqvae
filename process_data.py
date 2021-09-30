from environment.train_embeddings import get_model
from hparams import hparams


def ready_model():
    return get_model(**hparams, train_path="resources/full_dataset", data_depth=2)


def process_wave(wave, only_middle=False):
    model = ready_model()
    return None


def process_single_file(infile, outfile):
    wave = None  # read wave
    out_wave = process_wave(wave, only_middle=False)
    # save wave(out_wave, outfile)


def process_encoding_stream(input_stream):
    model = ready_model()
    yield None


def process_sound_stream(input_stream):
    model = ready_model()
    yield None


if __name__ == "__main__":
    process_single_file("input", "output")
