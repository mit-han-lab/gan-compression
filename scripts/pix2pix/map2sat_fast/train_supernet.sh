#!/usr/bin/env bash
python train_supernet.py --dataroot database/maps \
  --supernet resnet \
  --log_dir logs/pix2pix/map2sat_fast/supernet \
  --teacher_ngf 64 --student_ngf 64 --teacher_netG resnet_9blocks \
  --nepochs 200 --nepochs_decay 400 \
  --save_epoch_freq 50 --save_latest_freq 20000 \
  --eval_batch_size 16 \
  --restore_teacher_G_path pretrained/pix2pix/map2sat/full/latest_net_G.pth \
  --real_stat_path real_stat/maps_A.npz \
  --direction BtoA --config_set channels-64-pix2pix \
  --lambda_recon 10 --lambda_distill 0.01 --meta_path datasets/metas/maps/train1.meta
