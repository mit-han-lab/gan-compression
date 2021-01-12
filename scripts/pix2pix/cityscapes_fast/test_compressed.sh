#!/usr/bin/env bash
python test.py --dataroot database/cityscapes \
  --results_dir results-pretrained/pix2pix/cityscapes_fast/compressed \
  --config_str 16_24_48_56_40_48_24_16 \
  --restore_G_path pretrained/pix2pix/cityscapes_fast/compressed/latest_net_G.pth \
  --real_stat_path real_stat/cityscapes_A.npz \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/val_table.txt \
  --direction BtoA --need_profile
