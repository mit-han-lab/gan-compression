#!/usr/bin/env bash
python test.py --dataroot database/cityscapes \
  --results_dir results-pretrained/pix2pix/cityscapes/legacy \
  --config_str 48_32_40_40_24_32 \
  --restore_G_path pretrained/pix2pix/cityscapes/legacy/latest_net_G.pth \
  --real_stat_path real_stat/cityscapes_A.npz \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/table.txt \
  --direction BtoA --need_profile
