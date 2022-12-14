from hparams.parser import Hparams

default_forward_params = Hparams(
    n_fft=1024,
    hop_length=256,
    window_size=1024,
    sr=22050,
    channels=2,
    wav='',
    n_inps=1,
    n_hops=2,
    n_segment=1,
    n_total_segment=1,
    n_segment_each=1,
    prime_chunks=4,
    sample_length=0,
    sample_hop_length=30000,
    max_silence_pad_length=0,
    ignore_boundaries=False,
    use_nonrelative_specloss=True,
    multispec_loss_n_fft=(2048,1024,512),
    multispec_loss_hop_length=(240,120,50),
    multispec_loss_window_size=(1200,600,240),
    lmix_l1=0.0,
    lmix_l2=1.0,
    lmix_linf=0.02,
    linf_k=2048,
    bandwidth_cache="bandwidth.cache",
    bandwidth=None,
)
big_forward_params = default_forward_params.update(
    lmix_l1=0.02,
    lmix_l2=1.0,
    lmix_linf=0,
    sr=44100,
)
small_forward_params = default_forward_params.update(
    lmix_l1=0.0,
    lmix_l2=1.0,
    lmix_linf=0.02,
)
