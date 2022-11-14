from hparams.parser import Hparams

dirs_config = Hparams(
    restore_ckpt="best_model.ckpt",
    ckpt_dir='models',
    logs_dir="logs",
    default_ckpt_root="checkpoints",
)
