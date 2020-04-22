#!/usr/bin/env bash
python train.py --dataroot database/horse2zebra \
  --model cycle_gan \
  --netG resnet_9blocks \
  --log_dir logs/cycle_gan/horse2zebra/full \
  --real_stat_A_path real_stat/horse2zebra_A.npz \
  --real_stat_B_path real_stat/horse2zebra_B.npz
