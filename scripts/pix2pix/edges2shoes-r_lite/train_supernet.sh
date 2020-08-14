#!/usr/bin/env bash
python train_supernet.py --dataroot database/edges2shoes-r \
  --supernet resnet \
  --log_dir logs/pix2pix/edges2shoes-r_lite/supernet \
  --batch_size 4 --teacher_netG resnet_9blocks \
  --restore_teacher_G_path pretrained/pix2pix/edges2shoes-r/full/latest_net_G.pth \
  --real_stat_path real_stat/edges2shoes-r_B.npz \
  --nepochs 10 --nepochs_decay 30 \
  --teacher_ngf 64 --student_ngf 64 \
  --config_set channels-64-pix2pix

