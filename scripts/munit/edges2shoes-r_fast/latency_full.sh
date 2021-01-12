#!/usr/bin/env bash
python latency.py --dataroot database/edges2shoes-r-unaligned/valA \
  --dataset_mode single \
  --results_dir results-pretrained/munit/edges2shoes-r_fast/full/no-style \
  --restore_G_A_path pretrained/munit/edges2shoes-r/full/latest_net_G.pth \
  --need_profile --ngf 64 \
  --real_stat_path real_stat/edges2shoes-r_B.npz \
  --model munit_test --num_test 100 --netG munit
