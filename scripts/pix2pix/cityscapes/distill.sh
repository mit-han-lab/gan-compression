#!/usr/bin/env bash
python distill.py --dataroot database/cityscapes \
  --distiller resnet \
  --log_dir logs/pix2pix/cityscapes/distill \
  --direction BtoA \
  --nepochs 100 --nepochs_decay 150 \
  --save_latest_freq 25000 --save_epoch_freq 25 \
  --teacher_ngf 96 --pretrained_ngf 96 \
  --restore_teacher_G_path logs/pix2pix/cityscapes/mobile/checkpoints/latest_net_G.pth \
  --restore_pretrained_G_path logs/pix2pix/cityscapes/mobile/checkpoints/latest_net_G.pth \
  --restore_D_path logs/pix2pix/cityscapes/mobile/checkpoints/latest_net_D.pth \
  --real_stat_path real_stat/cityscapes_A.npz \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/val_table.txt --meta_path datasets/metas/cityscapes/train1.meta
