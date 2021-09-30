#dirty hack for tqdm/dataloaders bug
import sys
import tqdm
tqdm.auto = tqdm
sys.modules['tqdm.auto'] = tqdm
sys.modules['tqdm'] = tqdm
#end


from environment.train_embeddings import train
from hparams import hparams


def run():
    train(**hparams,
          #train_path="resources/full_musicnet/musicnet/musicnet/train_data",
          #test_path="resources/full_musicnet/musicnet/musicnet/test_data")
          train_path="resources/full_dataset", data_depth=2)


if __name__ == "__main__":
    run()
