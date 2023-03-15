#!/bin/bash
python3 run_training.py --gpus=[0,1] --config=config_big --model=diffusion --compressor=big_vae  --accelerator=ddp \
	--compressor.main_dir=generated/models/big_vae/big_vae_low_norm/ --train_path=resources/fma_wav \
	--diffusion.main_dir=generated/models/big_diffusion/fma_fixed_noise/ \
	--compressor.batch_size=10 --compressor.num_workers=4 --diffusion.prep_chunks=2 \
	--compressor.emb_width=4  --diffusion.opt_params.lr_decay=10000 \
	--diffusion.data_context_cond=0 --diffusion.condition_params.style_cond_size=0 \
	--diffusion.condition_params.listens_cond_size=0 \




