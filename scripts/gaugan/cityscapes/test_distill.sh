#!/usr/bin/env bash
python test.py --dataroot database/cityscapes-origin \
  --model spade --dataset_mode cityscapes \
  --results_dir results-pretrained/gaugan/cityscapes/distill \
  --ngf 48 --netG mobile_spade \
  --restore_G_path pretrained/gaugan/cityscapes/distill/latest_net_G.pth \
  --real_stat_path real_stat/cityscapes_A.npz \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/val_table.txt --need_profile