#!/usr/bin/env bash
python search.py --dataroot database/cityscapes \
  --restore_G_path logs/pix2pix/cityscapes/supernet/checkpoints/latest_net_G.pth \
  --output_path logs/pix2pix/cityscapes/supernet/results.pkl \
  --batch_size 4 --config_set channels-48 \
  --real_stat_path real_stat/cityscapes_A.npz \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/val_table.txt \
  --direction BtoA --no_fid --max_dataset_size 200
