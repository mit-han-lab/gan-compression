#!/usr/bin/env bash
python train.py --dataroot database/edges2shoes-r \
  --model pix2pix \
  --log_dir logs/pix2pix/edges2shoes-r_lite/mobile \
  --real_stat_path real_stat/edges2shoes-r_B.npz
