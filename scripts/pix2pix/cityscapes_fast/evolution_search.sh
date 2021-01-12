#!/usr/bin/env bash
python evolution_search.py --phase train --dataroot database/cityscapes \
  --restore_G_path logs/pix2pix/cityscapes_fast/supernet/checkpoints/latest_net_G.pth \
  --output_dir logs/pix2pix/cityscapes_fast/supernet/evolution \
  --batch_size 4 --ngf 64 --config_set channels-64-pix2pix \
  --real_stat_path real_stat/cityscapes_A.npz \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/train_table.txt \
  --direction BtoA --no_fid \
  --budget 5.7e9 --load_in_memory --criterion mIoU --mutate_prob 0.3 \
  --meta_path datasets/metas/cityscapes/train2.meta