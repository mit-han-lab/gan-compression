#!/usr/bin/env bash
python latency.py --dataroot database/cityscapes \
  --results_dir results-pretrained/pix2pix/cityscapes_lite/compressed \
  --config_str 32_32_48_32_64_40_16_32 \
  --restore_G_path pretrained/pix2pix/cityscapes_lite/compressed/latest_net_G.pth \
  --real_stat_path real_stat/cityscapes_A.npz \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/table.txt \
  --direction BtoA --need_profile
