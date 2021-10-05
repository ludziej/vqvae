import torch

from environment.train_embeddings import get_model
from hparams import hparams
import torchaudio
import numpy as np


def ready_model():
    return get_model(**hparams, train_path="resources/full_dataset", data_depth=2).to("cuda").eval()


def model_forward(model, wave, only_encoding=False, level=1):
    indata = wave.reshape(1, -1, 1).to("cuda")
    if only_encoding:
        return model.encode(indata, start_level=level, end_level=level+1)[0][0].to("cpu")
    x_out, loss, metrics = model(indata)
    return x_out.reshape(1, -1).repeat(1, 1).to("cpu")


def process_wave(wave, end_on=220416, start_from=0, only_encoding=False):
    model = ready_model()
    with torch.no_grad():
        out = model_forward(model, wave[start_from:end_on], only_encoding)
    return out


def code_acc(code1, code2):
    return torch.mean((code1 == code2).float())


def safe_load_mono(infile):
    sr = hparams["sr"]
    wave, file_sr = torchaudio.load(infile)
    assert file_sr == sr
    return (wave[0, :] + wave[1, :]) / 2


def process_single_file(infile, outfile=None, only_encoding=False, start_from=0, end_on=220416):
    wave = safe_load_mono(infile)
    out_wave = process_wave(wave, end_on=end_on, only_encoding=only_encoding, start_from=start_from)
    if only_encoding:
        return out_wave
    torchaudio.save(outfile, out_wave, hparams["sr"])


def process_stream(input_stream, suffix_size=1/4, to_numpy=False, only_encoding=False):
    model = ready_model()
    prev_suffix = None
    with torch.no_grad():
        for data in input_stream:
            data = torch.tensor(data).to("cuda")
            new_suff = data[-int(len(data)*suffix_size):]
            data = torch.cat([prev_suffix, data]) if prev_suffix is not None else data
            out = model_forward(model, data, only_encoding)
            to_remove = int(len(out)*suffix_size/(suffix_size+1))
            print("out = {}, to_remove = {}, returned = {}".format(out.shape, to_remove, out.shape[0] - to_remove))
            out = out[to_remove:] if prev_suffix is not None else out
            yield out.numpy() if to_numpy else out
            prev_suffix = new_suff


def check_accuracy_on_chunking(infile, chunk_size, chunks_number, suffix_size=1/4):
    wave = safe_load_mono(infile)
    out_code = process_wave(wave, end_on=chunk_size*chunks_number, only_encoding=True, start_from=0)
    chunked = (wave[i * chunk_size:(i + 1) * chunk_size] for i in range(chunks_number))
    out_chunk_coded = torch.cat(list(process_stream(chunked, suffix_size=suffix_size, only_encoding=True)))
    acc = code_acc(out_code, out_chunk_coded)
    print("Accuracy on chunking: {} ({} chunks of size {} with sr = {} and suffix_size={})"
          .format(acc, chunks_number, chunk_size, hparams["sr"], suffix_size))
    return out_code, out_chunk_coded


if __name__ == "__main__":
    #process_single_file("generated/input.wav", "generated/output.wav")
    #coded = process_single_file("generated/input.wav", None, only_encoding=True)
    #print(coded.shape)
    chunked, out_chunk_coded = check_accuracy_on_chunking("resources/cls_dataset/10.wav",
                                                          chunk_size=220416, chunks_number=4, suffix_size=1)
    pass
