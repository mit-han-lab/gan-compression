#!/usr/bin/env bash
python train_supernet.py --dataroot database/horse2zebra \
  --dataset_mode unaligned \
  --supernet resnet --teacher_netG resnet_9blocks \
  --log_dir logs/cycle_gan/horse2zebra_fast/supernet \
  --gan_mode lsgan \
  --student_ngf 64 --ndf 64 \
  --restore_teacher_G_path pretrained/cycle_gan/horse2zebra/full/latest_net_G.pth \
  --real_stat_path real_stat/horse2zebra_B.npz \
  --lambda_recon 10 --lambda_distill 0.01 \
  --nepochs 200 --nepochs_decay 200 \
  --save_epoch_freq 20 \
  --config_set channels-64-cycleGAN
