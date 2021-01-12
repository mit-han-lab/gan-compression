#!/usr/bin/env bash
python evolution_search.py --phase train --dataroot database/maps \
  --restore_G_path logs/pix2pix/map2sat_fast/supernet/checkpoints/latest_net_G.pth \
  --output_dir logs/pix2pix/map2sat_fast/supernet/evolution \
  --direction BtoA --batch_size 32 \
  --config_set channels-64-pix2pix --budget 4.6e9 \
  --real_stat_path real_stat/maps_subtrain_A.npz \
  --ngf 64 --mutate_prob 0.3 --meta_path datasets/metas/maps/train2.meta
