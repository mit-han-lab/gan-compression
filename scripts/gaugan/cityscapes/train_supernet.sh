#!/usr/bin/env bash
python train_supernet.py --dataroot database/cityscapes-origin \
  --supernet spade \
  --log_dir logs/gaugan/cityscapes/supernet \
  --restore_teacher_G_path logs/gaugan/cityscapes/mobile/checkpoint/latest_net_G.pth \
  --restore_student_G_path logs/gaugan/cityscapes/distill/checkpoints/latest_net_G.pth \
  --restore_D_path logs/gaugan/cityscapes/distill/checkpoints/latest_net_D.pth \
  --real_stat_path real_stat/cityscapes_A.npz \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/val_table.txt \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --load_in_memory --no_fid \
  --config_set channels-48 --meta_path datasets/metas/cityscapes-origin/train1.meta