#!/usr/bin/env bash
python evolution_search.py --phase train --dataroot database/cityscapes-origin \
  --model spade --netG super_mobile_spade \
  --dataset_mode cityscapes \
  --restore_G_path logs/gaugan/cityscapes_fast/supernet/checkpoints/latest_net_G.pth \
  --output_dir logs/gaugan/cityscapes_fast/supernet/evolution \
  --batch_size 16 --config_set channels-64 \
  --real_stat_path real_stat/cityscapes_A.npz \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/train_table.txt \
  --direction BtoA --no_fid \
  --budget 32e9 --ngf 64 --criterion mIoU \
  --meta_path datasets/metas/cityscapes-origin/train2.meta --weighted_sample 3 --weight_strategy geometric
