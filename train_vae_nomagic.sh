#!/bin/bash
python3 run_training.py --gpus=[0,1,2,3] --config=config_big --model=compressor --compressor=big_vae \
	--train_path=/scidata/mimuw-jan-ludziejewski/fma_full_wav \
	--compressor.main_dir=generated/models/big_vae/big_vae_high_emb/ --compressor.neptune_run_id="" \
	--compressor.batch_size=32 --compressor.num_workers=8 \
