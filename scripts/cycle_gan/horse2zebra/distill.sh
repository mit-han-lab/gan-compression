#!/usr/bin/env bash
python distill.py --dataroot database/horse2zebra \
--dataset_mode unaligned \
--distiller resnet \
--log_dir logs/cycle_gan/horse2zebra/distill \
--gan_mode lsgan \
--student_ngf 32 --ndf 64 \
--restore_teacher_G_path logs/cycle_gan/horse2zebra/mobile/checkpoints/latest_net_G_A.pth \
--restore_pretrained_G_path logs/cycle_gan/horse2zebra/mobile/checkpoints/latest_net_G_A.pth \
--restore_D_path logs/cycle_gan/horse2zebra/mobile/checkpoints/latest_net_D_A.pth \
--real_stat_path real_stat/horse2zebra_B.npz \
--lambda_recon 10 \
--lambda_distill 0.01 \
--nepochs 100 --nepochs_decay 100 \
--save_epoch_freq 20