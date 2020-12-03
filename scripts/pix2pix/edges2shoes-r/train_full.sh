#!/usr/bin/env bash
python train.py --dataroot database/edges2shoes-r \
  --model pix2pix --netG resnet_9blocks \
  --log_dir logs/pix2pix/edges2shoes-r/train \
  --real_stat_path real_stat/edges2shoes-r_B.npz \
  --batch_size 4