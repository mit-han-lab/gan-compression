#!/usr/bin/env bash
python evolution_search.py --dataroot database/cityscapes-origin \
  --model spade --netG super_mobile_spade \
  --dataset_mode cityscapes \
  --restore_G_path logs/gaugan/cityscapes_fast/supernet/checkpoints/latest_net_G.pth \
  --output_dir logs/gaugan/cityscapes_fast/evolution \
  --batch_size 16 --config_set channels-64 \
  --real_stat_path real_stat/cityscapes_A.npz \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/table.txt \
  --direction BtoA --no_fid \
  --budget 32e9 --ngf 64 --weighted_sample 10 --criterion mIoU