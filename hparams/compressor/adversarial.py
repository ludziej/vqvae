from hparams.parser import Hparams
import math


adv_params = Hparams(
    with_discriminator=False,
    adv_latency=2000,  # steps
    gan_loss_weight=0.1,
    disc_use_freq=1,
    disc_loss_weight=0.1,
    gan_loss_warmup=1000,
    discriminator_level=0,
    pooltype="avg",   # ["max", "avg"]
    reduce_type="avg",  # ["max", "avg"]
    type="joined",  # "mel", "wav", "fft" or "joined"
    n_mels=256,
    n_fft=2048,
    hop_length=240,
    window_size=1200,
    leaky=1e-2,
    res_depth=4,
    first_channels=32,
    trainable_prep=False,
    use_stride=True,
    use_amp=True,
    use_log_scale=True,
    n_bins=108,
    stop_disc_train_after=math.inf,
    first_double_downsample=0,
    multiply_level=False,
    pos_enc_size=0,
    classify_each_level=True,
)
