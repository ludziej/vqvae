from environment.train_embeddings import train as e_train
from environment.train_transformer import train_prior as p_train, train_upsampler as u_train
from environment.hparams_parser import HparamsParser
from hparams import hparams_dict, top_hparams


def run_trainer(hparams):
    train_fun = {"vqvae": e_train, "prior": p_train, "upsampler": u_train}
    return train_fun[hparams.model](hparams)


def run():
    hparams = HparamsParser(hparams_dict, top_hparams).create_hparams()
    run_trainer(hparams)


if __name__ == "__main__":
    run()
