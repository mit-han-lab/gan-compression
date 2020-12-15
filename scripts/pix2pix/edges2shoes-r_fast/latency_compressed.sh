#!/usr/bin/env bash
python latency.py --dataroot database/edges2shoes-r \
  --results_dir results-pretrained/pix2pix/edges2shoes-r_fast/compressed \
  --restore_G_path pretrained/pix2pix/edges2shoes-r_fast/compressed/latest_net_G.pth \
  --config_str 24_24_40_56_24_56_16_40 \
  --real_stat_path real_stat/edges2shoes-r_B.npz \
  --need_profile --num_test 500
