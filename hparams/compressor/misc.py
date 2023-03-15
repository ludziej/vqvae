from hparams.parser import Hparams

dirs_config = Hparams(
    restore_ckpt="last.ckpt",
    ckpt_dir='models',
    logs_dir="logs",
    distinct_ckpt_name='last_model',
    default_ckpt_root="checkpoints",
)
