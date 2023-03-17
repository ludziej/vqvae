#!/bin/bash
python3 run_training.py --gpus=[0] --config=config_big --model=compressor --compressor=big_vae \
  --train_path=resources/fma_wav \
  --compressor.main_dir=generated/models/big_vae/big_vae_high_emb/ --compressor.neptune_run_id="" \
	--compressor.batch_size=6 --compressor.num_workers=4 \



