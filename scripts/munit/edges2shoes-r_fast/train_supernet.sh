#!/usr/bin/env bash
python train_supernet.py --dataroot database/edges2shoes-r-unaligned \
  --supernet munit \
  --log_dir logs/munit/edges2shoes-r_fast/supernet \
  --real_stat_path real_stat/edges2shoes-r_B.npz \
  --restore_teacher_G_path pretrained/munit/edges2shoes-r_fast/full/latest_net_G.pth \
  --config_set channels-64 --nepochs 42 --niters 2000000 --lr_decay_steps 200000 \
  --student_netG super_mobile_munit \
  --metaA_path datasets/metas/edges2shoes-r-unaligned/train1A.meta \
  --metaB_path datasets/metas/edges2shoes-r-unaligned/train1B.meta
