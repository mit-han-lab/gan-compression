#!/usr/bin/env bash
python search.py --dataroot database/maps \
  --restore_G_path logs/pix2pix/map2sat/supernet/checkpoints/latest_net_G.pth \
  --output_path logs/pix2pix/map2sat/supernet/result.pkl \
  --direction BtoA --batch_size 32 \
  --config_set channels-48 \
  --real_stat_path real_stat/maps_A.npz
