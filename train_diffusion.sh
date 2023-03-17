#!/bin/bash
python3 run_training.py --gpus=[0,1] --config=config_big --model=diffusion --compressor=big_vae \
  --train_path=resources/fma_wav --diffusion.main_dir=generated/models/big_diffusion/fma_cond/ \
	--compressor.batch_size=10 --compressor.num_workers=4 --diffusion.prep_chunks=2 \
	--diffusion.condition_params.style_cond_size=256
