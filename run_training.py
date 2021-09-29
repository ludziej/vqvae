from environment.train_embeddings import train
from types import SimpleNamespace



forward_params = SimpleNamespace(
    n_fft=1024,
    hop_length=256,
    window_size=1024,
    sr=44100,
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
    lmix_l1=True,
    lmix_l2=True,
    lmix_linf=True,
    linf_k=2048,
    bandwidth={},
)

vqvaehparams = dict(
    levels = 2,
    loss_fn = "lmix",
    downs_t = (5, 3),
    strides_t = (2, 2),
    emb_width = 64,
    l_bins = 1024,
    l_mu = 0.99,
    commit = 0.02,
    spectral = 0.0,
    multispectral = 1.0,
    width = 32,
    depth = 4,
    m_conv = 1.0,
#    dilation_growth_rate = 3,
#    hvqvae_multipliers=(2, 1, 1),
#    lmix_l2=1.0,
#    lmix_linf=0.02,
    restore_vqvae='generated/jukebox/models/5b/vqvae.pth.tar',
    batch_size=2,
    sample_len=99840,
    num_workers=4,
    sr=22050,
    fp16=False,
    forward_params=forward_params,

)

bigvqvaehparams = dict(
    levels=3,
    downs_t=(3, 2, 2),
    strides_t=(2, 2, 2),
    emb_width=64,
    l_bins=2048,
    l_mu=0.99,
    commit=0.02,
    spectral=0.0,
    multispectral=1.0,
    hvqvae_multipliers=(2, 1, 1),
    loss_fn='lmix',
    lmix_l2=1.0,
    lmix_linf=0.02,
    width=32,
    depth=4,
    m_conv=1.0,
    dilation_growth_rate=3,
    restore_vqvae='generated/jukebox/models/5b/vqvae.pth.tar',
    batch_size=10,
    sample_len=100000,
    num_workers=4,
    sr=22050,
    fp16=False,
    forward_params=forward_params,
)


opt_hparams = dict(
    epochs=10000,
    lr=0.0003,
    clip=1.0,
    beta1=0.9,
    beta2=0.999,
    ignore_grad_norm=0,
    weight_decay=0.0,
    eps=1e-08,
    lr_warmup=100.0,
    lr_decay=10000000000.0,
    lr_gamma=1.0,
    lr_scale=1.0,
    lr_use_linear_decay=False,
    lr_start_linear_decay=0,
    lr_use_cosine_decay=False,
    fp16=False,
    fp16_params=False,
    fp16_loss_scale=None,
    fp16_scale_window=1000.0,
    fp16_opt=False,
    gpus=-1
)

hparams = {**opt_hparams, **vqvaehparams}


def run():
    train(**hparams,
          #train_path="resources/full_musicnet/musicnet/musicnet/train_data",
          #test_path="resources/full_musicnet/musicnet/musicnet/test_data")
          train_path="resources/full_dataset", data_depth=2)


run()
