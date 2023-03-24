from hparams.parser import HparamsParser
from hparams.config import hparams_registry


def run_trainer(hparams):
    if hparams.model == "compressor":
        from environment.train_embeddings import train as e_train
        e_train(hparams)
    elif hparams.model == "diffusion":
        from environment.train_diffusion import train as d_train
        d_train(hparams)
    elif hparams.model == "upsampler":
        from environment.train_transformer import train_upsampler as u_train
        u_train(hparams)
    elif hparams.model == "prior":
        from environment.train_transformer import train_prior as p_train
        p_train(hparams)
    else:
        raise ValueError(f"Unknown model to train: {hparams.model}")


def run():
    hparams = HparamsParser(hparams_registry).create_hparams()
    run_trainer(hparams)


if __name__ == "__main__":
    run()
