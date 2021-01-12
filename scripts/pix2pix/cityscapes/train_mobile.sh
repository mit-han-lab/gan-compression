#!/usr/bin/env bash
python train.py --dataroot database/cityscapes \
  --model pix2pix \
  --log_dir logs/pix2pix/cityscapes/mobile \
  --batch_size 1 \
  --save_latest_freq 25000 --save_epoch_freq 25 \
  --nepochs 100 --nepochs_decay 150 \
  --ngf 96 --direction BtoA \
  --real_stat_path real_stat/cityscapes_A.npz \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/val_table.txt
