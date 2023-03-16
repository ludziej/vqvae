#!/bin/bash
python3 run_training.py --gpus=[0] --config=config_big --model=compressor --train_path=resources/fma_wav --compressor=big_vae \
	--compressor.main_dir=generated/models/big_vae/big_vae_high_emb/  --accelerator=ddp --compressor.neptune_run_id="" \
	--compressor.batch_size=6 --compressor.num_workers=4 \
	--compressor.lr=0.001 --compressor.lr_decay=5000 --compressor.lr_gamma=0.7 --compressor.emb_width=16 \
	--compressor.bottleneck_lw=0.0001 --compressor.forward_params.lmix_l1=0.1 --compressor.forward_params.lmix_linf=0.1



