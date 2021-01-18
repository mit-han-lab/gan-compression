#!/usr/bin/env bash
python distill.py --dataroot database/maps \
  --distiller resnet \
  --log_dir logs/pix2pix/map2sat/distill \
  --teacher_ngf 96 --pretrained_ngf 96 \
  --nepochs 100 --nepochs_decay 200 \
  --save_epoch_freq 50 --save_latest_freq 20000 \
  --eval_batch_size 16 \
  --restore_teacher_G_path logs/pix2pix/map2sat/mobile/checkpoints/latest_net_G.pth \
  --restore_pretrained_G_path logs/pix2pix/map2sat/mobile/checkpoints/latest_net_G.pth \
  --restore_D_path logs/pix2pix/map2sat/mobile/checkpoints/latest_net_D.pth \
  --real_stat_path real_stat/maps_A.npz \
  --lambda_recon 10 --lambda_distill 0.01 \
  --direction BtoA --meta_path datasets/metas/maps/train1.meta
