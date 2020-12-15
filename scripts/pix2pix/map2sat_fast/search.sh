#!/usr/bin/env bash
python search.py --dataroot database/maps \
  --restore_G_path logs/pix2pix/map2sat_fast/supernet/checkpoints/latest_net_G.pth \
  --output_path logs/pix2pix/map2sat_fast/supernet/result.pkl \
  --direction BtoA --batch_size 32 \
  --config_set channels-64-pix2pix --budget 4.6e9 \
  --real_stat_path real_stat/maps_A.npz --ngf 64
