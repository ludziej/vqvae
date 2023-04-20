#!/bin/bash
python3 run_training.py --gpus=[0,1,2,3] --config=config_big --model=diffusion --compressor=big_vae \
	--train_path=/scidata/mimuw-jan-ludziejewski/fma_full_wav \
	--diffusion.main_dir=generated/models/big_diffusion/fma_unet_big_tok/ \
	--tags=[A100,attn_heads_8,max_width_2048,min_width_64,s_width_8,lr_decay_2k,block_depth_4] \
	--diffusion.autenc_params.attn_heads=8 --diffusion.autenc_params.width=8 --diffusion.autenc_params.max_width=2048 --diffusion.autenc_params.min_width=64 --diffusion.autenc_params.emb_width=2048 \
	--diffusion.opt_params.lr_decay=2000 --diffusion.autenc_params.depth=4 \	
	--compressor.batch_size=18 --compressor.num_workers=8 --diffusion.prep_chunks=2 \

