#!/bin/bash

# edit second and third line
python3 run_training.py --config=config_big --model=prior \
	--train_path=resources/fma_wav/ --gpus=[0,1] --compressor.main_dir=generated/models/big_vqvae/DRY_RUN/ \
	--compressor.num_workers=2 --compressor.batch_size=1 \
	--accelerator=ddp --logging=INFO  \
	--prior.main_dir=generated/models/big_prior/big_prior_A100/   \
	--prior.log_interval=5000 --prior.ckpt_freq=1000 --track_grad_norm=2 \
	--prior.dim=2048 --prior.token_dim=2048  --prior.depth=30 --prior.heads=8 --prior.dim_head=256 \
	--prior.lr=0.0005 --prior.opt_params.lr_decay=10000 --prior.opt_params.lr_warmup=1000 \
	--prior.rezero=1  --prior.feature_map_dims=128 \
	--prior.use_fasttransformer=0 --prior.prep_on_cpu=0 --prior.prepr_encode_chunks=3 \
	--prior.pos_init_scale=0.01 --prior.bins_init_scale=0.01 --prior.pos_enc_type=trainable --prior.restore_ckpt=last.ckpt
