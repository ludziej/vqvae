import logging

import torch

from environment.train_embeddings import get_model, WavCompressor
from hparams import hparams
import torchaudio
import numpy as np
from tqdm import tqdm


def ready_model() -> WavCompressor:
    return get_model(**hparams, train_path="resources/full_dataset", data_depth=2).to("cuda").eval()


def model_forward(model, wave, only_encoding=False, level=0):
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


def check_augment(infile, outfile=None, end_on=None, level=0):
    wave = safe_load_mono(infile)
    wave = wave[:end_on] if end_on is not None else wave
    model = ready_model()
    enc_signal = model_forward(model, wave, only_encoding=True, level=level).to('cuda').reshape(1, -1)
    aug_loss, aug_acc, out_wave, cl,  sig_var, aug_rss, var_loss, last_layer_acc, last_layer_usage = model.augmentation_is_close(wave.reshape(1, 1, -1), enc_signal, verbose=True)
    model.my_logger.info("Aug acc: {}, Aug loss: {}, Last Layer acc: {} Last layer usage: {}".format(aug_acc, aug_loss,last_layer_acc, last_layer_usage))
    if outfile is not None:
        torchaudio.save(outfile, out_wave[0], hparams["sr"])


def process_stream(input_stream, suffix_size=1/4, to_numpy=False, only_encoding=False):
    model = ready_model()
    prev_suffix = None
    with torch.no_grad():
        for data in input_stream:
            data = torch.tensor(data).to("cuda")
            in_data_len = len(data)
            suff_len = int(in_data_len*suffix_size)
            data = torch.cat([prev_suffix, data]) if prev_suffix is not None else data
            new_suff = data[-suff_len:]
            out = model_forward(model, data, only_encoding)
            to_remove = int(len(out)*(len(data) - in_data_len)/len(data))
            model.my_logger.info("out = {}, to_remove = {}, returned = {}".format(out.shape, to_remove, out.shape[0] - to_remove))
            out = out[to_remove:] if prev_suffix is not None else out
            yield out.numpy() if to_numpy else out
            prev_suffix = new_suff


def check_accuracy_on_chunking(infile, chunk_size, chunks_number, suffix_size=1/4):
    wave = safe_load_mono(infile)
    out_code = process_wave(wave, end_on=chunk_size*chunks_number, only_encoding=True, start_from=0)
    chunked = (wave[i * chunk_size:(i + 1) * chunk_size] for i in range(chunks_number))
    out_chunk_coded = torch.cat(list(process_stream(chunked, suffix_size=suffix_size, only_encoding=True)))
    print("Chunked cat size = {}, one push size = {}".format(out_chunk_coded.shape, out_code.shape))
    acc = code_acc(out_code, out_chunk_coded)
    print("Accuracy on chunking: {} ({} chunks of size {} with sr = {} ({} sec) and suffix_size={})"
          .format(acc, chunks_number, chunk_size, hparams["sr"], chunk_size / hparams["sr"], suffix_size))
    return out_code, out_chunk_coded


def find_working_chunk_size(input, start_i, chunks, suffix_size):
    i = start_i
    while i:
        try:
            chunked, out_chunk_coded =\
                check_accuracy_on_chunking(input, chunk_size=i, chunks_number=chunks, suffix_size=suffix_size)
            break
        except Exception as e:
            print("{} did not work - {}".format(i, e))
            i -= chunks
    print("Finally worked - {}".format(i))
    return i


if __name__ == "__main__":
    working_chunk_05_s = 11250
    working_chunk_02_s = 5625

    #out = process_single_file("resources/full_dataset/webern_symphony/webern_symphony_ref_1.wav", "generated/output.wav", only_encoding=True)
    #coded = process_single_file("generated/input.wav", None, only_encoding=True)
    #logging.info(coded.shape)
                                                          #chunk_size=220416//3//2, chunks_number=4, suffix_size=2)
    for i in range(40):
        print(i)
        check_augment("resources/cls_dataset/10.wav",  "generated/out_aug{}.wav".format(i), end_on=150500)
    #find_working_chunk_size("resources/cls_dataset/10.wav", start_i=working_chunk_05_s, chunks=10, suffix_size=4)
    #find_working_chunk_size("resources/cls_dataset/10.wav", start_i=working_chunk_05_s, chunks=10, suffix_size=10)
    #find_working_chunk_size("resources/cls_dataset/10.wav", start_i=working_chunk_02_s, chunks=10, suffix_size=4)
    #find_working_chunk_size("resources/cls_dataset/10.wav", start_i=working_chunk_02_s, chunks=50, suffix_size=10)
    pass
