#!/usr/bin/env bash
python evolution_search.py --dataroot database/horse2zebra/trainA \
  --dataset_mode single --phase train \
  --restore_G_path logs/cycle_gan/horse2zebra_fast/supernet/checkpoints/latest_net_G.pth \
  --output_dir logs/cycle_gan/horse2zebra_fast/supernet/evolution \
  --ngf 64 --batch_size 32 \
  --config_set channels-64-cycleGAN --mutate_prob 0.4 \
  --real_stat_path real_stat/horse2zebra_B.npz --budget 3e9 \
  --weighted_sample 2 --meta_path datasets/metas/horse2zebra/train2A.meta
