#!/usr/bin/env bash
python latency.py --dataroot database/cityscapes \
  --results_dir results-pretrained/pix2pix/cityscapes/full \
  --ngf 64 --netG resnet_9blocks \
  --restore_G_path pretrained/pix2pix/cityscapes/full/latest_net_G.pth \
  --real_stat_path real_stat/cityscapes_A.npz \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/val_table.txt \
  --direction BtoA --need_profile
