#!/usr/bin/env bash
python train.py --dataroot database/edges2shoes-r-unaligned \
  --model munit \
  --log_dir logs/munit/edges2shoes-r_fast/full \
  --real_stat_A_path real_stat/edges2shoes-r_A.npz \
  --real_stat_B_path real_stat/edges2shoes-r_B.npz
