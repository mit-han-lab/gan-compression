#!/usr/bin/env bash
python search.py --dataroot database/edges2shoes-r \
  --restore_G_path logs/pix2pix/edges2shoes-r_fast/supernet/checkpoints/latest_net_G.pth \
  --output_path logs/pix2pix/edges2shoes-r_fast/supernet/result.pkl \
  --ngf 64 --batch_size 32 \
  --config_set channels-64-pix2pix \
  --real_stat_path real_stat/edges2shoes-r_B.npz --load_in_memory --budget 6e9
