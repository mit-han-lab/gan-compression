#!/usr/bin/env bash
python train_supernet.py --dataroot database/edges2shoes-r \
  --supernet resnet \
  --log_dir logs/pix2pix/edges2shoes-r_lite/supernet-stage1 \
  --batch_size 4 \
  --restore_teacher_G_path logs/pix2pix/edges2shoes-r_lite/mobile/checkpoints/latest_net_G.pth \
  --restore_student_G_path logs/pix2pix/edges2shoes-r_lite/mobile/checkpoints/latest_net_G.pth \
  --restore_D_path logs/pix2pix/edges2shoes-r_lite/mobile/checkpoints/latest_net_G.pth \
  --real_stat_path real_stat/edges2shoes-r_B.npz \
  --nepochs 2 --nepochs_decay 8 \
  --teacher_ngf 64 --student_ngf 64 \
  --config_set channels-64-pix2pix-stage1 --sort_channels

python train_supernet.py --dataroot database/edges2shoes-r \
  --supernet resnet \
  --log_dir logs/pix2pix/edges2shoes-r_lite/supernet-stage2 \
  --batch_size 4 \
  --restore_teacher_G_path logs/pix2pix/edges2shoes-r_lite/mobile/checkpoints/latest_net_G.pth \
  --restore_student_G_path logs/pix2pix/edges2shoes-r_lite/supernet-stage1/checkpoints/latest_net_G.pth \
  --restore_D_path logs/pix2pix/edges2shoes-r_lite/supernet-stage1/checkpoints/latest_net_G.pth \
  --real_stat_path real_stat/edges2shoes-r_B.npz \
  --nepochs 10 --nepochs_decay 30 \
  --teacher_ngf 64 --student_ngf 64 \
  --config_set channels-64-pix2pix
