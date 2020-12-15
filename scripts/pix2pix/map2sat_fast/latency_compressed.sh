#!/usr/bin/env bash
python latency.py --dataroot database/maps \
  --results_dir results-pretrained/pix2pix/map2sat_fast/compressed \
  --restore_G_path pretrained/pix2pix/map2sat_fast/compressed/latest_net_G.pth \
  --config_str 16_16_40_40_40_64_24_16 \
  --real_stat_path real_stat/maps_A.npz \
  --direction BtoA \
  --need_profile --num_test 200
