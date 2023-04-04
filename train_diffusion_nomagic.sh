#!/bin/bash
python3 run_training.py --gpus=[0,1,2,3] --config=config_big --model=diffusion --compressor=big_vae \
	--train_path=/scidata/mimuw-jan-ludziejewski/fma_full_wav \
	--diffusion.main_dir=generated/models/big_diffusion/fma_nonorm/ --diffusion.neptune_run_id="" \
	--compressor.batch_size=18 --compressor.num_workers=8 --diffusion.prep_chunks=2 \

