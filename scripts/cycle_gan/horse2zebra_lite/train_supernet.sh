#!/usr/bin/env bash
python train_supernet.py --dataroot database/horse2zebra \
  --dataset_mode unaligned \
  --supernet resnet \
  --log_dir logs/cycle_gan/horse2zebra_lite/supernet-stage1 \
  --gan_mode lsgan \
  --student_ngf 64 --ndf 64 \
  --restore_teacher_G_path logs/pix2pix/horse2zebra_lite/mobile/checkpoints/latest_net_G.pth \
  --restore_student_G_path logs/pix2pix/horse2zebra_lite/mobile/checkpoints/latest_net_G.pth \
  --restore_D_path logs/pix2pix/horse2zebra_lite/mobile/checkpoints/checkpoints/latest_net_D.pth \
  --real_stat_path real_stat/horse2zebra_B.npz \
  --lambda_recon 10 --lambda_distill 0.01 \
  --nepochs 50 --nepochs_decay 50 \
  --save_epoch_freq 20 \
  --config_set channels-64-cycleGAN-stage1 --sort_channels
python train_supernet.py --dataroot database/horse2zebra \
  --dataset_mode unaligned \
  --supernet resnet \
  --log_dir logs/cycle_gan/horse2zebra_lite/supernet-stage2 \
  --gan_mode lsgan \
  --student_ngf 64 --ndf 64 \
  --restore_teacher_G_path logs/pix2pix/horse2zebra_lite/mobile/checkpoints/latest_net_G.pth \
  --restore_student_G_path logs/cycle_gan/horse2zebra_lite/supernet-stage1/checkpoints/latest_net_G.pth \
  --restore_D_path logs/cycle_gan/horse2zebra_lite/supernet-stage2/checkpoints/latest_net_D.pth \
  --real_stat_path real_stat/horse2zebra_B.npz \
  --lambda_recon 10 --lambda_distill 0.01 \
  --nepochs 200 --nepochs_decay 200 \
  --save_epoch_freq 20 \
  --config_set channels-64-cycleGAN
