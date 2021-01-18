#!/usr/bin/env bash
python search.py --phase train --dataroot database/cityscapes-origin \
  --model spade --netG super_mobile_spade \
  --dataset_mode cityscapes \
  --restore_G_path logs/gaugan/cityscapes/supernet/checkpoints/latest_net_G.pth \
  --output_path logs/gaugan/cityscapes/supernet/results.pkl \
  --batch_size 16 --config_set channels-48 \
  --real_stat_path real_stat/cityscapes_A.npz \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/train_table.txt \
  --direction BtoA --no_fid --budget 35e9 \
  --meta_path datasets/metas/cityscapes-origin/train2.meta