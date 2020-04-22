#!/usr/bin/env bash
python train_supernet.py --dataroot database/cityscapes \
  --supernet resnet \
  --log_dir logs/pix2pix/cityscapes/supernet \
  --restore_teacher_G_path logs/pix2pix/cityscapes/mobile/checkpoints/latest_net_G.pth \
  --restore_student_G_path logs/pix2pix/cityscapes/distill/checkpoints/latest_net_G.pth \
  --restore_D_path logs/pix2pix/cityscapes/distill/checkpoints/latest_net_D.pth \
  --real_stat_path real_stat/cityscapes_A.npz \
  --nepochs 160 --nepochs_decay 240 \
  --save_latest_freq 25000 --save_epoch_freq 25 \
  --teacher_ngf 96 \
  --config_set channels-48 \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/table.txt \
  --eval_batch_size 2 \
  --direction BtoA
