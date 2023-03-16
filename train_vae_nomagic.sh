#!/bin/bash
python3 run_training.py --gpus=[0,1,2,3,4,5,6,7] --config=config_big --model=compressor --train_path=/scidata/mimuw-jan-ludziejewski/fma_full_wav --compressor=big_vae \
	--compressor.main_dir=generated/models/big_vae/big_vae_low_norm/ --compressor.restore_ckpt=last.ckpt \
	--compressor.log_interval=500 --track_grad_norm=0 --compressor.ckpt_freq=500 --compressor.logger_type="neptune" \
	--compressor.batch_size=32 --compressor.num_workers=8 --compressor.sample_len=262144 --compressor.use_audiofile=0 --accelerator=ddp \
       	--compressor.lr=0.001 --compressor.lr_decay=5000 --compressor.lr_gamma=0.7 --compressor.emb_width=4 \
	--compressor.bottleneck_lw=0.0001 --track_grad_norm=-1  \
	--compressor.forward_params.lmix_l1=0.1 --compressor.forward_params.lmix_linf=0.1
