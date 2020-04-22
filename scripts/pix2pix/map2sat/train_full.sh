#!/usr/bin/env bash
python train.py --dataroot database/maps \
  --model pix2pix \
  --log_dir logs/pix2pix/map2sat/full \
  --netG resnet_9blocks \
  --batch_size 1 \
  --lambda_recon 10 \
  --nepochs 100 --nepochs_decay 200 \
  --save_epoch_freq 50 --save_latest_freq 20000 \
  --eval_batch_size 16 --real_stat_path real_stat/maps_A.npz \
  --direction BtoA
