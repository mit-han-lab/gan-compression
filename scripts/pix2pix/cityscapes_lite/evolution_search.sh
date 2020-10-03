#!/usr/bin/env bash
python evolution_search.py --dataroot database/cityscapes \
  --restore_G_path logs/pix2pix/cityscapes_lite/supernet/checkpoints/latest_net_G.pth \
  --output_dir logs/pix2pix/cityscapes_lite/supernet/evolution \
  --batch_size 4 --ngf 64 --config_set channels-64-pix2pix \
  --real_stat_path real_stat/cityscapes_A.npz \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/table.txt \
  --direction BtoA --no_fid \
  --budget 5.7e9 --load_in_memory --weighted_sample 2 --criterion mIoU
