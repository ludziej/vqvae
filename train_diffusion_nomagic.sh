#!/bin/bash
python3 run_training.py --gpus=[0,1,2,3] --config=config_big --model=diffusion --train_path=/scidata/mimuw-jan-ludziejewski/fma_full_wav --compressor=big_vae \
        --log_every_n_steps=200 \
        --diffusion.autenc_params.log_weights_norm=0 --diffusion.autenc_params.use_log_grads=0 \
	--compressor.main_dir=generated/models/big_vae/big_vae_low_norm/ --compressor.restore_ckpt=last.ckpt \
	--track_grad_norm=-1 --diffusion.ckpt_freq=1000 --device_stats=0 --compressor.emb_width=4 \
        --compressor.batch_size=18 --compressor.num_workers=8 --compressor.use_audiofile=0 --accelerator=ddp \
        --diffusion.n_ctx=10240 --diffusion.log_interval=500 --diffusion.prep_chunks=2 \
        --diffusion.main_dir=generated/models/big_diffusion/fma_new_vae/ --diffusion.restore_ckpt=last.ckpt \
        --diffusion.opt_params.lr=0.0001 --diffusion.opt_params.lr_warmup=5000 --diffusion.prep_level=1 \
	--diffusion.diff_params.noise_steps=1000 --diffusion.opt_params.lr_decay=10000 --diffusion.opt_params.lr_gamma=0.7 \
        --diffusion.autenc_params.emb_width=2048 --diffusion.autenc_params.width=128 --diffusion.autenc_params.depth=8 \
        --diffusion.autenc_params.downs_t=[4] --diffusion.autenc_params.dilation_growth_rate=2 \
        --diffusion.log_sample_bs=2 --diffusion.max_logged_sounds=2 \
        --diffusion.autenc_params.norm_type=group --diffusion.autenc_params.use_weight_standard=1 \
        --diffusion.diff_params.beta_start=0.0015 --diffusion.diff_params.beta_end=0.0195 \
        --diffusion.diff_params.renormalize_sampling=0 \
        --diffusion.autenc_params.skip_connections_step=1 --diffusion.autenc_params.channel_increase=2 \
        --diffusion.rmse_loss_weight=0 --diffusion.eps_loss_weight=1 \
        --diffusion.autenc_params.bottleneck_type=none --diffusion.autenc_params.leaky_param=0.01 \
        --diffusion.autenc_params.bottleneck_params.depth=2 --diffusion.autenc_params.bottleneck_params.ff_mul=2 \
        --diffusion.autenc_params.dilation_cycle=4 --diffusion.autenc_params.res_scale=1. \
	--diffusion.autenc_params.rezero=1 --diffusion.autenc_params.rezero_in_attn=1 \
        --diffusion.data_context_cond=0 --diffusion.condition_params.style_cond_size=0 \
        --diffusion.condition_params.listens_cond_size=0 \



