#!/usr/bin/env bash
python search.py --dataroot database/cityscapes-origin \
  --model spade --netG super_mobile_spade \
  --dataset_mode cityscapes \
  --restore_G_path logs/gaugan/cityscapes_lite/supernet/checkpoints/latest_net_G.pth \
  --output_path logs/gaugan/cityscapes_lite/supernet/results.pkl \
  --batch_size 16 --config_set channels-64 \
  --real_stat_path real_stat/cityscapes_A.npz \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/table.txt \
  --direction BtoA --no_fid --max_dataset_size 200 \
  --budget 35e9 --ngf 64