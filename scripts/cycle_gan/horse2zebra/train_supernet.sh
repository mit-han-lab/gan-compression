#!/usr/bin/env bash
python train_supernet.py --dataroot database/horse2zebra \
  --dataset_mode unaligned \
  --supernet resnet \
  --log_dir logs/cycle_gan/horse2zebra/supernet \
  --gan_mode lsgan \
  --student_ngf 32 --ndf 64 \
  --restore_teacher_G_path logs/cycle_gan/horse2zebra/mobile/checkpoints/latest_net_G_A.pth \
  --restore_student_G_path logs/cycle_gan/horse2zebra/distill/checkpoints/latest_net_G.pth \
  --restore_D_path logs/cycle_gan/horse2zebra/distill/checkpoints/latest_net_D.pth \
  --real_stat_path real_stat/horse2zebra_B.npz \
  --lambda_recon 10 --lambda_distill 0.01 \
  --nepochs 200 --nepochs_decay 200 \
  --save_epoch_freq 20 \
  --config_set channels-32
