#!/usr/bin/env bash
python train.py --dataroot database/maps \
  --model pix2pix \
  --log_dir logs/pix2pix/map2sat/mobile \
  --batch_size 1 \
  --ngf 96 --lambda_recon 10 \
  --nepochs 100 --nepochs_decay 200 \
  --save_epoch_freq 50 --save_latest_freq 20000 \
  --eval_batch_size 16 --real_stat_path real_stat/maps_A.npz \
  --direction BtoA
