#!/usr/bin/env bash
python train_supernet.py --dataroot database/cityscapes \
  --supernet resnet \
  --log_dir logs/pix2pix/cityscapes_lite/supernet \
  --restore_teacher_G_path pretrained/pix2pix/cityscapes/full/latest_net_G.pth \
  --real_stat_path real_stat/cityscapes_A.npz \
  --nepochs 200 --nepochs_decay 300 \
  --save_latest_freq 25000 --save_epoch_freq 25 \
  --teacher_ngf 64 --student_ngf 64 --teacher_netG resnet_9blocks \
  --config_set channels-64-pix2pix \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/table.txt \
  --eval_batch_size 2 \
  --direction BtoA
