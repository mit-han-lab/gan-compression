#!/usr/bin/env bash
python3 train.py --dataroot database/fomm \
  --model pix2pix \
  --dataset_mode triplet\
  --log_dir logs/pix2pix/fomm/2_mobile_nocoord_motion_wandb \
  --real_stat_path real_stat/fomm_B.npz \
  --batch_size 32 \
  --ngf 96 \
  --lambda_recon 10 \
  --nepochs 200 \
  --nepochs_decay 200 \
  --save_epoch_freq 50 \
  --save_latest_freq 20000 \
  --eval_batch_size 16 \
  --num_threads 0 \
  --use_wandb \
  --use_motion 
