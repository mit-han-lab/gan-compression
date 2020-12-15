#!/usr/bin/env bash
python search.py --dataroot database/horse2zebra/valA \
  --dataset_mode single \
  --restore_G_path logs/cycle_gan/horse2zebra_fast/supernet/checkpoints/latest_net_G.pth \
  --output_path logs/cycle_gan/horse2zebra_fast/supernet/result.pkl \
  --ngf 64 --batch_size 32 \
  --config_set channels-64-cycleGAN \
  --real_stat_path real_stat/horse2zebra_B.npz --budget 3.6e9
