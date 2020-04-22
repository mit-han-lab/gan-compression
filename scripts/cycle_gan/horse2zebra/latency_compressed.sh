#!/usr/bin/env bash
python latency.py --dataroot database/horse2zebra/valA \
  --dataset_mode single \
  --results_dir results-pretrained/cycle_gan/horse2zebra/compressed \
  --config_str 16_16_32_16_32_32_16_16 \
  --restore_G_path pretrained/cycle_gan/horse2zebra/compressed/latest_net_G.pth \
  --need_profile \
  --real_stat_path real_stat/horse2zebra_B.npz
